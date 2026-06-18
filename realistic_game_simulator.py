"""
Simulador de Partits amb Quintets Reals i Rotacions Basades en Minuts
=====================================================================

Aquest simulador usa els quintets reals que han jugat junts durant
la temporada i els rota segons els minuts que han jugat junts.

Exemple: Si un equip té 3 quintets:
- Quintet A: 800 minuts (50%)
- Quintet B: 500 minuts (31%)
- Quintet C: 300 minuts (19%)

Durant un partit de 48 minuts, jugaran aproximadament:
- Quintet A: 24 minuts
- Quintet B: 15 minuts
- Quintet C: 9 minuts
"""

import numpy as np
import pandas as pd
import json
import random
from typing import Dict, List, Tuple
from complete_markov_model import *


class RealisticLineupRotation:
    """
    Gestiona les rotacions realistes d'un equip basat en minuts reals.
    """
    
    def __init__(self, team_name: str, lineups_data: List[Dict],
                 player_data_df: pd.DataFrame, max_lineups: int = 20,
                 player_minutes: Dict[str, float] = None,
                 all_lineups_df: pd.DataFrame = None):
        """
        Args:
            team_name: Nom de l'equip
            lineups_data: Llista de quintets amb minuts del JSON
            player_data_df: DataFrame amb dades dels jugadors
            max_lineups: Màxim de quintets a usar (per defecte 20, els més usats)
            player_minutes: OPCIONAL - Dict amb minuts individuals {player_name: minutes}
            all_lineups_df: OPCIONAL - DataFrame amb tots els lineups del CSV per matching
        """
        self.team_name = team_name
        self.player_df = player_data_df
        # NOU: Guardar lineups_df per passar-lo a cada AdvancedLineupFromData
        self.all_lineups_df = all_lineups_df

        # MODE 1: Si tenim minuts individuals, fer assignació intel·ligent
        if player_minutes is not None and all_lineups_df is not None:
            print(f"\n   🧠 Mode intel·ligent: assignant {len(player_minutes)} jugadors a lineups reals")
            lineups_data = self._allocate_minutes_to_real_lineups(
                player_minutes, team_name, all_lineups_df
            )
        
        # Ordenar per minuts (més minuts primer) i limitar
        lineups_sorted = sorted(lineups_data, key=lambda x: x['minutes'], reverse=True)
        
        if max_lineups and len(lineups_sorted) > max_lineups:
            print(f"   ℹ️  Limitant a top {max_lineups} quintets (de {len(lineups_sorted)} totals)")
            self.lineups_data = lineups_sorted[:max_lineups]
        else:
            self.lineups_data = lineups_sorted
        
        # Calcular distribució de minuts (només dels quintets seleccionats)
        self.total_minutes = sum(l['minutes'] for l in self.lineups_data)
        
        # Crear quintets de Markov
        self.markov_lineups = []
        self.minute_distribution = []
        
        for lineup_info in self.lineups_data:
            # PRIORITAT: Extreure IDs del lineup_id
            player_ids = self._extract_ids_from_lineup_id(lineup_info.get('lineup_id', ''))
            
            if len(player_ids) == 5:
                # Crear quintet usant IDs directament del lineup_id
                players = self._create_players_from_ids(player_ids)
            else:
                # Fallback: si no hi ha lineup_id vàlid, intentar amb noms
                print(f"   ⚠️  lineup_id no vàlid, usant noms com fallback")
                players = self._create_players_from_names(lineup_info['players'])
                player_ids = []
            
            if len(players) == 5:
                lineup = AdvancedLineupFromData(
                    f"{team_name} - Lineup {len(self.markov_lineups)+1}",
                    players,
                    lineups_df=self.all_lineups_df  # NOU: passar CSV per calibració Bayesiana
                )
                
                # IMPORTANT: Obtenir PACE del quintet
                # Prioritat 1: Del JSON si existeix
                if 'pace' in lineup_info and lineup_info['pace'] and lineup_info['pace'] > 0:
                    lineup.pace = lineup_info['pace']
                # Prioritat 2: Calcular des de stats si estan disponibles
                elif all(k in lineup_info for k in ['FGA', 'FTA', 'OREB', 'TOV', 'GP', 'MIN']):
                    try:
                        # Fórmula: Pace = 48 × (FGA + 0.44×FTA - OREB + TOV) / GP / (MIN/GP)
                        possessions_per_game = (
                            lineup_info['FGA'] + 
                            0.44 * lineup_info['FTA'] - 
                            lineup_info['OREB'] + 
                            lineup_info['TOV']
                        ) / lineup_info['GP']
                        
                        minutes_per_game = lineup_info['MIN'] / lineup_info['GP']
                        pace = (possessions_per_game / minutes_per_game) * 48
                        
                        # Limitar a valors NBA realistes
                        lineup.pace = np.clip(pace, 90, 110)
                    except:
                        lineup.pace = 100.0  # Default si falla
                else:
                    # Prioritat 3: Estimar des de offensive rating i punts
                    if 'OFF_RATING' in lineup_info and lineup_info['OFF_RATING'] > 0:
                        # OFF_RATING ≈ (PTS / Poss) × 100
                        # Típicament OFF_RATING entre 100-120, això dóna pace entre 95-105
                        estimated_pace = 90 + (lineup_info['OFF_RATING'] - 100) * 0.5
                        lineup.pace = np.clip(estimated_pace, 90, 110)
                    else:
                        lineup.pace = 100.0  # Default NBA mitjana
                
                self.markov_lineups.append({
                    'lineup': lineup,
                    'minutes': lineup_info['minutes'],
                    'pct': lineup_info['minutes'] / self.total_minutes,
                    'net_rating': lineup_info.get('net_rating', 0),
                    'pace': lineup.pace,  # Guardar també aquí
                    'player_names': [p.name for p in players],  # Noms reals dels jugadors carregats
                    'player_ids': player_ids
                })
                
                self.minute_distribution.append(lineup_info['minutes'] / self.total_minutes)
            else:
                print(f"   ⚠️  Quintet incomplet: només {len(players)}/5 jugadors trobats")
        
        # Normalitzar distribució
        total_pct = sum(self.minute_distribution)
        if total_pct > 0:
            self.minute_distribution = [p / total_pct for p in self.minute_distribution]
        
        print(f"✓ {team_name}: {len(self.markov_lineups)} quintets carregats ({self.total_minutes:.0f} min totals)")
        
        # Mostrar info dels top quintets
        if len(self.markov_lineups) > 0:
            top_lineup = self.markov_lineups[0]
            print(f"   Top lineup: {top_lineup['minutes']:.0f} min ({top_lineup['pct']*100:.1f}%) - Net: {top_lineup['net_rating']:+.1f}")
    
    def _extract_ids_from_lineup_id(self, lineup_id: str) -> List[int]:
        """
        Extreu els IDs dels jugadors del lineup_id.
        
        Args:
            lineup_id: Ex: "-203497-203944-1628978-1630162-1630183-"
            
        Returns:
            [203497, 203944, 1628978, 1630162, 1630183]
        """
        if not lineup_id:
            return []
        
        # Eliminar guions i separar
        ids_str = lineup_id.strip('-').split('-')
        
        # Convertir a int
        ids = []
        for id_str in ids_str:
            try:
                player_id = int(id_str)
                if player_id > 0:  # IDs vàlids són positius
                    ids.append(player_id)
            except ValueError:
                continue
        
        return ids
    
    def _create_players_from_ids(self, player_ids: List[int]) -> List[AdvancedPlayerFromData]:
        """Crea objectes jugador des d'IDs (mètode preferit)."""
        players = []
        
        for player_id in player_ids:
            try:
                # Buscar jugador per ID
                matches = self.player_df[self.player_df['player_id'] == player_id]
                
                if len(matches) > 0:
                    player_data = matches.iloc[0].to_dict()
                    player = AdvancedPlayerFromData(player_data)
                    players.append(player)
                else:
                    print(f"   ⚠️  Jugador no trobat amb ID: {player_id}")
            except Exception as e:
                print(f"   ⚠️  Error creant jugador amb ID {player_id}: {e}")
        
        return players
    
    def _create_players_from_names(self, player_names: List[str]) -> List[AdvancedPlayerFromData]:
        """Crea objectes jugador des de noms (fallback)."""
        players = []
        
        for name in player_names:
            try:
                # Buscar jugador al DataFrame
                matches = self.player_df[
                    self.player_df['player_name'].str.contains(name, case=False, na=False)
                ]
                
                if len(matches) > 0:
                    player_data = matches.iloc[0].to_dict()
                    player = AdvancedPlayerFromData(player_data)
                    players.append(player)
                else:
                    # Jugador no trobat - usar per defecte
                    print(f"   ⚠️  Jugador no trobat: {name}")
            except Exception as e:
                print(f"   ⚠️  Error creant {name}: {e}")
        
        return players
    
    def _allocate_minutes_to_real_lineups(self, player_minutes: Dict[str, float], 
                                          team_name: str, all_lineups_df: pd.DataFrame) -> List[Dict]:
        """
        MÈTODE INTEL·LIGENT AMB IDS: Assigna minuts individuals als quintets reals
        
        Usa player_id per matching exacte (evita problemes amb cognoms duplicats)
        
        Args:
            player_minutes: Dict amb {player_name: minutes}
            team_name: Nom/abreviació de l'equip (ex: 'DAL', 'MIN')
            all_lineups_df: DataFrame amb tots els lineups del CSV
            
        Returns:
            List de lineups amb minuts assignats
        """
        
        # MAPA D'ABREVIACIONS → NOMS COMPLETS
        team_name_map = {
            'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
            'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
            'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
            'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
            'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
            'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
            'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
            'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
            'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
            'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
        }
        
        team_full_name = team_name_map.get(team_name, team_name)
        
        # 1. CONVERTIR NOMS DE JUGADORS → IDs
        print(f"      🔍 Convertint noms a IDs...")
        player_ids_to_minutes = {}  # {player_id: minutes}
        player_names_to_ids = {}    # {player_name: player_id}
        
        for player_name, minutes in player_minutes.items():
            # Buscar aquest jugador al CSV de jugadors per obtenir el seu ID
            player_match = self.player_df[
                (self.player_df['player_name'] == player_name) &
                (self.player_df['team_name'] == team_name)
            ]
            
            if len(player_match) == 0:
                # Provar amb cognom
                last_name = player_name.split()[-1]
                player_match = self.player_df[
                    (self.player_df['player_name'].str.contains(last_name, case=False, na=False)) &
                    (self.player_df['team_name'] == team_name)
                ]
            
            if len(player_match) > 0:
                player_id = str(player_match.iloc[0]['player_id'])
                player_ids_to_minutes[player_id] = minutes
                player_names_to_ids[player_name] = player_id
        
        if len(player_ids_to_minutes) < 5:
            print(f"      ⚠️  Només {len(player_ids_to_minutes)} jugadors amb ID trobats")
            return self._create_generic_lineups_from_minutes(player_minutes)
        
        print(f"      ✓ {len(player_ids_to_minutes)} jugadors amb ID")
        
        # 2. Filtrar lineups d'aquest equip
        team_lineups = all_lineups_df[
            all_lineups_df['TEAM_NAME'].str.contains(team_full_name, case=False, na=False)
        ].copy()
        
        if len(team_lineups) == 0:
            team_lineups = all_lineups_df[
                all_lineups_df['TEAM_NAME'].str.contains(team_name, case=False, na=False)
            ].copy()
        
        if len(team_lineups) == 0:
            print(f"      ⚠️  No hi ha lineups al CSV per {team_name}")
            return self._create_generic_lineups_from_minutes(player_minutes)
        
        # 3. Trobar quintets que inclouen els jugadors (COMPARANT IDs)
        matching_lineups = []
        
        for idx, lineup_row in team_lineups.iterrows():
            # Extreure IDs del GROUP_ID (ex: "-203497-203944-1628978-1630162-1630183-")
            if 'GROUP_ID' in lineup_row and pd.notna(lineup_row['GROUP_ID']):
                group_id = str(lineup_row['GROUP_ID'])
                # Separar per '-' i eliminar buits
                lineup_player_ids = [pid.strip() for pid in group_id.split('-') if pid.strip()]
            else:
                continue
            
            if len(lineup_player_ids) != 5:
                continue
            
            # Comprovar quants dels jugadors del box score estan en aquest quintet
            matching_count = sum(1 for pid in lineup_player_ids if pid in player_ids_to_minutes)
            
            if matching_count >= 3:  # Almenys 3/5 jugadors
                # Obtenir noms dels jugadors per debug
                player_names_in_lineup = []
                if 'GROUP_NAME' in lineup_row:
                    player_names_in_lineup = [p.strip() for p in str(lineup_row['GROUP_NAME']).split('-')]
                
                matching_lineups.append({
                    'player_ids': lineup_player_ids,
                    'player_names': player_names_in_lineup,  # Per debug
                    'matching_count': matching_count,
                    'net_rating': lineup_row.get('NET_RATING', 0),
                    'win_pct': lineup_row.get('W_PCT', 0.5)
                })
        
        if len(matching_lineups) == 0:
            print(f"      ⚠️  Cap quintet coincideix per IDs")
            return self._create_generic_lineups_from_minutes(player_minutes)
        
        # Ordenar per coincidències i net rating
        matching_lineups.sort(key=lambda x: (x['matching_count'], x['net_rating']), reverse=True)
        
        print(f"      ✓ {len(matching_lineups)} quintets reals trobats")
        
        # ═══════════════════════════════════════════════════════════════════
        # 4. ALGORISME GREEDY DE COBERTURA COMPLETA
        # ═══════════════════════════════════════════════════════════════════
        # OBJECTIU: assignar TOTS els minuts disponibles a quintets, de manera
        # que cada jugador acabi amb 0 minuts restants.
        # ESTRATÈGIA:
        #   1. Mentre quedin minuts per assignar:
        #        a. Trobar el quintet REAL amb millor cobertura (5/5 jugadors disponibles)
        #        b. Si no n'hi ha cap 5/5, buscar 4/5 (i completar amb jugador restant)
        #        c. Si no n'hi ha cap 4/5, generar quintet sintètic amb jugadors top
        #        d. Assignar tants minuts com sigui possible
        # ═══════════════════════════════════════════════════════════════════
        remaining_minutes = player_ids_to_minutes.copy()
        selected_lineups = []
        total_minutes_to_assign = sum(remaining_minutes.values())

        print(f"      📊 Total minuts a assignar: {total_minutes_to_assign:.0f}")

        # Crear mapa invers ID → nom per buscar noms ràpidament
        id_to_name = {pid: name for name, pid in player_names_to_ids.items()}

        iteration = 0
        max_iterations = 100  # Augmentat per garantir cobertura completa

        while sum(remaining_minutes.values()) > 0.5 and iteration < max_iterations:
            iteration += 1

            # ─── Pas 1: Buscar el millor quintet REAL ───────────────────────
            # Prioritzem quintets amb 5/5 jugadors disponibles, ordenats per
            # coverage total i net_rating
            best_lineup = None
            best_score = -1
            best_match_count = 0

            for lineup in matching_lineups:
                # Comptar quants jugadors d'aquest quintet encara tenen minuts
                available_count = sum(
                    1 for pid in lineup['player_ids']
                    if remaining_minutes.get(pid, 0) > 0.5
                )

                if available_count == 0:
                    continue

                # Coverage = suma dels minuts restants dels jugadors del quintet
                coverage = sum(
                    remaining_minutes.get(pid, 0)
                    for pid in lineup['player_ids']
                )

                # Score: prioritzar match_count alt, després coverage, després net_rating
                score = available_count * 1000 + coverage + lineup['net_rating'] * 0.5

                if score > best_score:
                    best_score = score
                    best_lineup = lineup
                    best_match_count = available_count

            # ─── Pas 2: Calcular els minuts assignables al quintet ──────────
            if best_lineup is not None and best_match_count >= 5:
                # Quintet 5/5 perfecte: minuts = mínim dels 5 jugadors
                lineup_ids = best_lineup['player_ids']
                lineup_minute_values = [
                    remaining_minutes.get(pid, 0)
                    for pid in lineup_ids
                ]
                # Limitar al mínim per evitar que un jugador entri en negatiu
                lineup_minutes = min(lineup_minute_values)

                player_names_for_lineup = [
                    id_to_name.get(pid, f"Player_{pid}")
                    for pid in lineup_ids
                ]
                lineup_id_final = '-'.join(lineup_ids)
                lineup_type = "real"
                net_rating = best_lineup.get('net_rating', 0)

            elif best_lineup is not None and best_match_count >= 3:
                # Quintet parcial (3-4/5): completar amb jugadors restants amb minuts
                lineup_ids = list(best_lineup['player_ids'])
                # Identificar jugadors del quintet que JA NO tenen minuts
                missing_slots = [
                    i for i, pid in enumerate(lineup_ids)
                    if remaining_minutes.get(pid, 0) <= 0.5
                ]

                # Trobar jugadors amb més minuts restants per omplir slots
                players_available = sorted(
                    [(pid, mins) for pid, mins in remaining_minutes.items() if mins > 0.5],
                    key=lambda x: -x[1]
                )

                # Substituir els jugadors sense minuts
                used_ids = set(lineup_ids)
                replacement_idx = 0
                for slot_idx in missing_slots:
                    # Buscar un jugador no usat amb minuts disponibles
                    while replacement_idx < len(players_available):
                        candidate_pid, _ = players_available[replacement_idx]
                        replacement_idx += 1
                        if candidate_pid not in used_ids:
                            lineup_ids[slot_idx] = candidate_pid
                            used_ids.add(candidate_pid)
                            break
                    else:
                        break  # No queden candidats

                # Calcular minuts amb el nou quintet
                lineup_minute_values = [
                    remaining_minutes.get(pid, 0)
                    for pid in lineup_ids
                ]
                lineup_minutes = min(lineup_minute_values)

                if lineup_minutes <= 0.5:
                    # No es pot assignar res, marcar aquest quintet com a esgotat
                    matching_lineups = [l for l in matching_lineups if l != best_lineup]
                    continue

                player_names_for_lineup = [
                    id_to_name.get(pid, f"Player_{pid}")
                    for pid in lineup_ids
                ]
                lineup_id_final = '-'.join(lineup_ids)
                lineup_type = "parcial"
                net_rating = best_lineup.get('net_rating', 0) * 0.5  # Penalitzar

            else:
                # No hi ha cap quintet adequat: crear quintet sintètic amb top jugadors
                players_available = sorted(
                    [(pid, mins) for pid, mins in remaining_minutes.items() if mins > 0.5],
                    key=lambda x: -x[1]
                )

                if len(players_available) < 5:
                    # Menys de 5 jugadors amb minuts: assignar tot el restant
                    # repartit equitativament entre els que queden
                    if len(players_available) == 0:
                        break
                    # Crear un últim "quintet" amb els jugadors que queden
                    lineup_ids = [pid for pid, _ in players_available]
                    # Repetir l'últim per arribar a 5 si cal (per al constructor)
                    while len(lineup_ids) < 5:
                        lineup_ids.append(players_available[0][0])

                    lineup_minutes = min(mins for _, mins in players_available)
                    player_names_for_lineup = [
                        id_to_name.get(pid, f"Player_{pid}")
                        for pid in lineup_ids
                    ]
                    lineup_id_final = '-'.join(lineup_ids)
                    lineup_type = "sintètic-residual"
                    net_rating = 0
                else:
                    # 5+ jugadors disponibles: agafar els 5 amb més minuts
                    lineup_ids = [pid for pid, _ in players_available[:5]]
                    lineup_minute_values = [mins for _, mins in players_available[:5]]
                    lineup_minutes = min(lineup_minute_values)

                    player_names_for_lineup = [
                        id_to_name.get(pid, f"Player_{pid}")
                        for pid in lineup_ids
                    ]
                    lineup_id_final = '-'.join(lineup_ids)
                    lineup_type = "sintètic"
                    net_rating = 0

            # ─── Pas 3: Afegir el quintet seleccionat ───────────────────────
            if lineup_minutes <= 0.5:
                # Defensa contra bucle infinit
                break

            selected_lineups.append({
                'players': player_names_for_lineup,
                'minutes': float(lineup_minutes),
                'lineup_id': lineup_id_final,
                'net_rating': net_rating,
                'type': lineup_type  # Per al debug
            })

            # ─── Pas 4: Reduir minuts restants ──────────────────────────────
            for player_id in lineup_ids:
                if player_id in remaining_minutes:
                    remaining_minutes[player_id] = max(0, remaining_minutes[player_id] - lineup_minutes)

        # ═══════════════════════════════════════════════════════════════════
        # 5. RESUM I IMPRESSIÓ DELS LINEUPS ASSIGNATS
        # ═══════════════════════════════════════════════════════════════════
        total_assigned = sum(l['minutes'] for l in selected_lineups)
        uncovered = sum(remaining_minutes.values())
        coverage_pct = (total_assigned / total_minutes_to_assign * 100) if total_minutes_to_assign > 0 else 0

        print(f"      ✅ {len(selected_lineups)} quintets assignats")
        print(f"      📊 Minuts coberts: {total_assigned:.0f} / {total_minutes_to_assign:.0f} ({coverage_pct:.1f}%)")
        if uncovered > 0.5:
            print(f"      ⚠️  Minuts sense cobrir: {uncovered:.1f}")

        # Detall dels lineups creats
        print(f"\n      📋 LINEUPS ASSIGNATS AL PARTIT:")
        for i, lineup in enumerate(selected_lineups, 1):
            type_marker = {"real": "✓", "parcial": "~", "sintètic": "★", "sintètic-residual": "✗"}.get(
                lineup.get('type', 'real'), "?"
            )
            players_str = ", ".join(lineup['players'])
            print(f"      {type_marker} Lineup {i}: {lineup['minutes']:.1f} min")
            print(f"         {players_str}")

        return selected_lineups if selected_lineups else self._create_generic_lineups_from_minutes(player_minutes)
        """
        MÈTODE INTEL·LIGENT: Assigna minuts individuals als quintets reals
        
        Args:
            player_minutes: Dict amb {player_name: minutes}
            team_name: Nom/abreviació de l'equip (ex: 'DAL', 'MIN')
            all_lineups_df: DataFrame amb tots els lineups del CSV
            
        Returns:
            List de lineups amb minuts assignats
        """
        
        # MAPA D'ABREVIACIONS → NOMS COMPLETS
        # El box score retorna abreviacions (DAL, MIN) però el CSV té noms complets
        team_name_map = {
            'ATL': 'Atlanta Hawks',
            'BOS': 'Boston Celtics',
            'BKN': 'Brooklyn Nets',
            'CHA': 'Charlotte Hornets',
            'CHI': 'Chicago Bulls',
            'CLE': 'Cleveland Cavaliers',
            'DAL': 'Dallas Mavericks',
            'DEN': 'Denver Nuggets',
            'DET': 'Detroit Pistons',
            'GSW': 'Golden State Warriors',
            'HOU': 'Houston Rockets',
            'IND': 'Indiana Pacers',
            'LAC': 'Los Angeles Clippers',
            'LAL': 'Los Angeles Lakers',
            'MEM': 'Memphis Grizzlies',
            'MIA': 'Miami Heat',
            'MIL': 'Milwaukee Bucks',
            'MIN': 'Minnesota Timberwolves',
            'NOP': 'New Orleans Pelicans',
            'NYK': 'New York Knicks',
            'OKC': 'Oklahoma City Thunder',
            'ORL': 'Orlando Magic',
            'PHI': 'Philadelphia 76ers',
            'PHX': 'Phoenix Suns',
            'POR': 'Portland Trail Blazers',
            'SAC': 'Sacramento Kings',
            'SAS': 'San Antonio Spurs',
            'TOR': 'Toronto Raptors',
            'UTA': 'Utah Jazz',
            'WAS': 'Washington Wizards'
        }
        
        # Convertir abreviació a nom complet
        team_full_name = team_name_map.get(team_name, team_name)
        
        # 1. Filtrar lineups d'aquest equip
        team_lineups = all_lineups_df[
            all_lineups_df['TEAM_NAME'].str.contains(team_full_name, case=False, na=False)
        ].copy()
        
        if len(team_lineups) == 0:
            # Provar amb l'abreviació directament per si de cas
            team_lineups = all_lineups_df[
                all_lineups_df['TEAM_NAME'].str.contains(team_name, case=False, na=False)
            ].copy()
        
        if len(team_lineups) == 0:
            print(f"      ⚠️  No hi ha lineups al CSV per {team_name} ({team_full_name})")
            return self._create_generic_lineups_from_minutes(player_minutes)
        
        # 2. Trobar quintets que inclouen els jugadors del partit
        matching_lineups = []
        
        for idx, lineup_row in team_lineups.iterrows():
            # Extreure noms jugadors
            if 'GROUP_NAME' in lineup_row:
                lineup_players = [p.strip() for p in str(lineup_row['GROUP_NAME']).split('-')]
            else:
                lineup_players = [lineup_row.get(f'PLAYER{i}', '') for i in range(1, 6)]
            
            lineup_players = [p for p in lineup_players if p and isinstance(p, str)]
            
            if len(lineup_players) != 5:
                continue
            
            # Comprovar coincidències comparant INICIAL + COGNOM
            # CSV: "K. Durant", Box Score: "Kevin Durant"
            # Solució: K + Durant vs K + Durant
            matching_count = 0
            
            for lineup_player in lineup_players:
                # Del CSV: "K. Durant" → inicial "K" + cognom "Durant"
                parts = lineup_player.split()
                if len(parts) < 2:
                    continue
                
                csv_initial = parts[0][0].upper()  # Primera lletra (K)
                csv_last_name = parts[-1].lower()   # Últim mot (durant)
                
                # Buscar aquest inicial+cognom als jugadors del box score
                for box_player in player_minutes.keys():
                    box_parts = box_player.split()
                    if len(box_parts) < 2:
                        continue
                    
                    box_initial = box_parts[0][0].upper()  # K de Kevin
                    box_last_name = box_parts[-1].lower()  # durant
                    
                    # Comparar inicial + cognom
                    if csv_initial == box_initial and csv_last_name == box_last_name:
                        matching_count += 1
                        break
            
            if matching_count >= 3:  # Almenys 3/5 jugadors coincideixen
                matching_lineups.append({
                    'players': lineup_players,
                    'matching_count': matching_count,
                    'net_rating': lineup_row.get('NET_RATING', 0),
                    'win_pct': lineup_row.get('W_PCT', 0.5)
                })
        
        if len(matching_lineups) == 0:
            print(f"      ⚠️  Cap quintet coincideix, creant genèrics")
            return self._create_generic_lineups_from_minutes(player_minutes)
        
        # Ordenar per coincidències i net rating
        matching_lineups.sort(
            key=lambda x: (x['matching_count'], x['net_rating']), 
            reverse=True
        )
        
        print(f"      ✓ {len(matching_lineups)} quintets reals trobats")
        
        # Debug: Mostrar total de minuts a assignar
        total_to_assign = sum(player_minutes.values())
        print(f"      📊 Total minuts a assignar: {total_to_assign:.0f}")
        
        # 3. GREEDY ALLOCATION: Assignar minuts als quintets
        remaining_minutes = player_minutes.copy()
        selected_lineups = []
        
        iteration = 0
        max_iterations = 15  # Augmentat per assegurar cobertura completa
        
        while sum(remaining_minutes.values()) > 1 and iteration < max_iterations:  # Continuar fins gairebé 0
            iteration += 1
            
            # Trobar millor quintet per cobrir minuts restants
            best_lineup = None
            best_score = 0
            
            for lineup in matching_lineups:
                # Calcular cobertura comparant inicial + cognom
                coverage = 0
                for lineup_player in lineup['players']:
                    parts = lineup_player.split()
                    if len(parts) < 2:
                        continue
                    
                    csv_initial = parts[0][0].upper()
                    csv_last_name = parts[-1].lower()
                    
                    # Buscar aquest inicial+cognom als jugadors amb minuts restants
                    for box_player, minutes in remaining_minutes.items():
                        box_parts = box_player.split()
                        if len(box_parts) < 2:
                            continue
                        
                        box_initial = box_parts[0][0].upper()
                        box_last_name = box_parts[-1].lower()
                        
                        if csv_initial == box_initial and csv_last_name == box_last_name:
                            coverage += minutes
                            break
                
                score = coverage + lineup['net_rating'] * 0.5
                
                if score > best_score:
                    best_score = score
                    best_lineup = lineup
            
            if best_lineup is None:
                break
            
            # IMPORTANT: Calcular minuts d'aquest quintet
            # = MÍNIM dels minuts restants dels jugadors que estan al quintet
            lineup_minute_values = []
            for lineup_player in best_lineup['players']:
                parts = lineup_player.split()
                if len(parts) < 2:
                    continue
                
                csv_initial = parts[0][0].upper()
                csv_last_name = parts[-1].lower()
                
                # Buscar aquest inicial+cognom als jugadors amb minuts
                for box_player, minutes in remaining_minutes.items():
                    box_parts = box_player.split()
                    if len(box_parts) < 2:
                        continue
                    
                    box_initial = box_parts[0][0].upper()
                    box_last_name = box_parts[-1].lower()
                    
                    if csv_initial == box_initial and csv_last_name == box_last_name and minutes > 0:
                        lineup_minute_values.append(minutes)
                        break
            
            lineup_minutes = min(lineup_minute_values) if lineup_minute_values else 0
            
            if lineup_minutes <= 0:
                # Si cap jugador del quintet té minuts, provar següent
                matching_lineups.remove(best_lineup)
                continue
            
            # Crear lineup_id
            lineup_id = self._make_lineup_id(best_lineup['players'])
            
            selected_lineups.append({
                'players': best_lineup['players'],
                'minutes': float(lineup_minutes),
                'lineup_id': lineup_id,
                'net_rating': best_lineup.get('net_rating', 0)
            })
            
            # REDUIR minuts restants de TOTS els jugadors del quintet
            for lineup_player in best_lineup['players']:
                parts = lineup_player.split()
                if len(parts) < 2:
                    continue
                
                csv_initial = parts[0][0].upper()
                csv_last_name = parts[-1].lower()
                
                # Buscar aquest inicial+cognom als jugadors del box score
                for box_player in list(remaining_minutes.keys()):
                    box_parts = box_player.split()
                    if len(box_parts) < 2:
                        continue
                    
                    box_initial = box_parts[0][0].upper()
                    box_last_name = box_parts[-1].lower()
                    
                    if csv_initial == box_initial and csv_last_name == box_last_name:
                        remaining_minutes[box_player] = max(0, remaining_minutes[box_player] - lineup_minutes)
                        break
        
        # IMPORTANT: Convertir noms dels quintets de format CSV (C. Flagg) a format complet (Cooper Flagg)
        # Això és necessari perquè el _create_players_from_names busca els noms complets
        for lineup in selected_lineups:
            full_name_players = []
            for csv_player_name in lineup['players']:
                parts = csv_player_name.split()
                if len(parts) < 2:
                    full_name_players.append(csv_player_name)
                    continue
                
                csv_initial = parts[0][0].upper()
                csv_last_name = parts[-1].lower()
                
                # Buscar el nom complet al box score comparant inicial+cognom
                full_name_found = None
                for box_player_name in player_minutes.keys():
                    box_parts = box_player_name.split()
                    if len(box_parts) < 2:
                        continue
                    
                    box_initial = box_parts[0][0].upper()
                    box_last_name = box_parts[-1].lower()
                    
                    if csv_initial == box_initial and csv_last_name == box_last_name:
                        full_name_found = box_player_name
                        break
                
                # Usar nom complet si es troba, sinó mantenir el del CSV
                full_name_players.append(full_name_found if full_name_found else csv_player_name)
            
            lineup['players'] = full_name_players
        
        # VERIFICACIÓ CRÍTICA: Assegurar que tots els minuts s'han cobert
        uncovered_total = sum(remaining_minutes.values())
        
        if uncovered_total > 1:  # Si encara falten més d'1 minut
            print(f"      ⚠️  {uncovered_total:.1f} min sense cobrir, creant quintets addicionals")
            
            # Afegir quintets amb jugadors restants
            players_remaining = [(p, m) for p, m in remaining_minutes.items() if m > 1]
            
            while len(players_remaining) > 0 and sum(m for p, m in players_remaining) > 1:
                # Crear quintet amb els 5 jugadors amb més minuts restants
                players_remaining.sort(key=lambda x: -x[1])
                
                # Usar noms COMPLETS del box score (no del CSV)
                quintet_players = [p for p, m in players_remaining[:5]]
                
                # Completar fins a 5 si cal
                while len(quintet_players) < 5:
                    # Afegir jugador que ja ha jugat tots els seus minuts
                    for p in player_minutes.keys():
                        if p not in quintet_players:
                            quintet_players.append(p)
                            break
                
                if len(quintet_players) < 5:
                    break
                
                # Calcular minuts: mínim dels jugadors del quintet
                quintet_minutes = min([remaining_minutes.get(p, 0) for p in quintet_players])
                
                if quintet_minutes <= 0:
                    break
                
                selected_lineups.append({
                    'players': quintet_players,  # Ja són noms complets!
                    'minutes': float(quintet_minutes),
                    'lineup_id': self._make_lineup_id(quintet_players)
                })
                
                # Actualitzar minuts restants
                for p in quintet_players:
                    if p in remaining_minutes:
                        remaining_minutes[p] = max(0, remaining_minutes[p] - quintet_minutes)
                
                # Actualitzar llista
                players_remaining = [(p, m) for p, m in remaining_minutes.items() if m > 1]
        
        # VERIFICACIÓ FINAL
        final_uncovered = sum(remaining_minutes.values())
        if final_uncovered > 2:
            print(f"      ❌ ATENCIÓ: {final_uncovered:.1f} min encara sense cobrir!")
            for p, m in remaining_minutes.items():
                if m > 1:
                    print(f"         • {p}: {m:.1f} min restants")
        
        print(f"      ✅ {len(selected_lineups)} quintets assignats ({sum(l['minutes'] for l in selected_lineups):.0f} min totals)")
        
        return selected_lineups if selected_lineups else self._create_generic_lineups_from_minutes(player_minutes)
    
    def _create_generic_lineups_from_minutes(self, player_minutes: Dict[str, float]) -> List[Dict]:
        """Crea lineups genèriques quan no hi ha quintets reals"""
        
        players_sorted = sorted(player_minutes.items(), key=lambda x: -x[1])
        
        # Starters (top 5)
        starters = players_sorted[:5]
        starter_minutes = sum(m for p, m in starters)
        
        lineups = [{
            'players': [p for p, m in starters],
            'minutes': starter_minutes,
            'lineup_id': self._make_lineup_id([p for p, m in starters])
        }]
        
        # Bench
        if len(players_sorted) > 5:
            bench = players_sorted[5:10]
            while len(bench) < 5:
                bench.append(starters[-1])
            
            bench_minutes = sum(m for p, m in bench[:5])
            lineups.append({
                'players': [p for p, m in bench[:5]],
                'minutes': bench_minutes,
                'lineup_id': self._make_lineup_id([p for p, m in bench[:5]])
            })
        
        return lineups
    
    def _make_lineup_id(self, player_names: List[str]) -> str:
        """Genera lineup_id pels jugadors"""
        ids = []
        for name in player_names:
            matches = self.player_df[
                self.player_df['player_name'] == name
            ]
            if len(matches) > 0:
                ids.append(str(matches.iloc[0]['player_id']))
            else:
                ids.append('0')
        return '-'.join(ids)
    
    def get_lineup_for_minute(self, game_minute: float) -> AdvancedLineupFromData:
        """
        Retorna el quintet que hauria d'estar jugant en aquest minut.
        Usa distribució de probabilitat basada en minuts reals.
        
        Args:
            game_minute: Minut del partit (0-48)
            
        Returns:
            Quintet de Markov
        """
        if len(self.markov_lineups) == 0:
            raise ValueError(f"No hi ha quintets per {self.team_name}")
        
        # Decidir quin quintet basant-se en probabilitats
        # Més minuts reals = més probabilitat de jugar
        lineup_idx = np.random.choice(
            len(self.markov_lineups),
            p=self.minute_distribution
        )
        
        return self.markov_lineups[lineup_idx]['lineup']
    
    def get_lineup_for_period(self, period_start: float, period_end: float) -> AdvancedLineupFromData:
        """
        Retorna el quintet que hauria de jugar durant un període.
        
        Args:
            period_start: Minut d'inici
            period_end: Minut de final
            
        Returns:
            Quintet de Markov
        """
        # Usar el punt mig del període
        mid_point = (period_start + period_end) / 2
        return self.get_lineup_for_minute(mid_point)
    
    def show_rotation_plan(self, game_minutes: int = 48):
        """Mostra el pla de rotació previst."""
        print(f"\n📋 PLA DE ROTACIÓ: {self.team_name}")
        print(f"{'='*70}\n")
        
        for i, lineup_info in enumerate(self.markov_lineups, 1):
            expected_minutes = game_minutes * lineup_info['pct']
            pace = lineup_info.get('pace', 100.0)
            
            print(f"{i}. {', '.join(lineup_info['player_names'][:3])}...")
            print(f"   Minuts esperats: {expected_minutes:.1f} ({lineup_info['pct']*100:.1f}%)")
            print(f"   Net Rating: {lineup_info['net_rating']:+.1f}")
            print(f"   Pace: {pace:.1f} poss/48min")
            
            # Mostrar IDs si estan disponibles
            if lineup_info.get('player_ids'):
                print(f"   IDs: {lineup_info['player_ids']}")
            
            print()


class RealisticGameSimulator:
    """
    Simula un partit complet amb quintets reals i rotacions realistes.
    """
    
    def __init__(self, team_a_rotation: RealisticLineupRotation,
                 team_b_rotation: RealisticLineupRotation,
                 home_team: str = 'a',
                 home_court_pts: float = 2.8):
        """
        Args:
            team_a_rotation: Rotació de l'equip A
            team_b_rotation: Rotació de l'equip B
            home_team: Quin equip juga a casa: 'a', 'b' o None (camp neutral).
                       Per defecte 'a' (l'equip A és el local).
            home_court_pts: Magnitud de l'avantatge local en punts (NBA ≈ 2.5-3.0).
        """
        self.team_a = team_a_rotation
        self.team_b = team_b_rotation

        # ──────────────────────────────────────────────────────────────────
        # MILLORA A: AVANTATGE DE PISTA LOCAL
        # En comptes de sumar punts plans al final (que no afecta la dinàmica
        # del partit), donem a l'equip local una mica més d'eficiència ofensiva
        # i un pèl més de pace. Repartim ~80% via eficiència i ~20% via pace,
        # que és com es manifesta el home-court advantage real.
        # ──────────────────────────────────────────────────────────────────
        self.home_team = home_team
        self.home_court_pts = home_court_pts

        NBA_AVG_PTS = 113.0  # punts mitjans per equip i partit
        self.home_off_boost = 1.0 + 0.8 * (home_court_pts / NBA_AVG_PTS)
        self.home_pace_boost = 1.0 + 0.2 * (home_court_pts / NBA_AVG_PTS)

        # Aplicar el factor ofensiu local UNA sola vegada als quintets de
        # l'equip local (els objectes es reutilitzen entre simulacions, així
        # que NO s'ha de multiplicar dins del bucle per evitar acumulació).
        self._apply_home_court_factor()

    def _apply_home_court_factor(self):
        """MILLORA A: fixa home_court_factor als quintets segons qui és local.

        markov_lineups guarda cada quintet com un dict {'lineup': <obj>, ...},
        així que accedim a l'objecte real sota la clau 'lineup'. També és
        tolerant si en alguna versió s'hi guarden objectes directament.
        """
        boost_a = self.home_off_boost if self.home_team == 'a' else 1.0
        boost_b = self.home_off_boost if self.home_team == 'b' else 1.0

        def set_boost(entries, boost):
            for entry in entries:
                # Cas normal: dict amb l'objecte sota 'lineup'
                if isinstance(entry, dict):
                    obj = entry.get('lineup')
                else:
                    # Cas alternatiu: l'objecte directament
                    obj = entry
                if obj is not None and hasattr(obj, 'matrix'):
                    obj.home_court_factor = boost

        set_boost(getattr(self.team_a, 'markov_lineups', []), boost_a)
        set_boost(getattr(self.team_b, 'markov_lineups', []), boost_b)
    
    def simulate_game(self, verbose: bool = False, 
                     show_quarters: bool = True,
                     show_key_moments: bool = True) -> Dict:
        """
        Simula un partit complet de 48 minuts NBA.
        Cada equip té possessions segons el seu PACE individual.
        
        Args:
            verbose: Mostrar cada possessió
            show_quarters: Mostrar resum per quarters
            show_key_moments: Mostrar només moments clau (runs, lead changes)
            
        Returns:
            Diccionari amb resultats detallats
        """
        # Constants NBA
        GAME_MINUTES = 48.0
        QUARTER_MINUTES = 12.0
        
        # Inicialitzar
        score_a = 0
        score_b = 0
        possession_log = []
        quarter_scores = {1: {'a': 0, 'b': 0}, 2: {'a': 0, 'b': 0}, 
                         3: {'a': 0, 'b': 0}, 4: {'a': 0, 'b': 0}}
        
        # Stats avançades
        lead_changes = 0
        ties = 0
        biggest_lead_a = 0
        biggest_lead_b = 0
        current_run_a = 0
        current_run_b = 0
        longest_run_a = 0
        longest_run_b = 0
        
        total_possessions_a = 0
        total_possessions_b = 0
        
        # Tracking de pace per segment
        pace_samples_a = []
        pace_samples_b = []
        
        if verbose or show_key_moments:
            print(f"\n{'='*70}")
            print(f"🏀 {self.team_a.team_name} vs {self.team_b.team_name}")
            print(f"{'='*70}\n")
        
        # NUEVO: Calcular possessions totals per cada equip segons el seu pace
        # Simulem minut a minut, però cada equip pot tenir diferent nombre de possessions
        
        current_minute = 0.0
        quarter = 1
        quarter_start_a = 0
        quarter_start_b = 0
        quarter_poss_a = 0
        quarter_poss_b = 0
        
        # Pool de possessions pendents per cada equip
        pending_possessions_a = 0.0
        pending_possessions_b = 0.0
        
        if show_quarters:
            print(f"\n{'─'*70}")
            print(f"📊 QUARTER {quarter}")
            print(f"{'─'*70}")
        
        while current_minute < GAME_MINUTES:
            # Obtenir quintets actius segons el minut actual
            lineup_a = self.team_a.get_lineup_for_minute(current_minute)
            lineup_b = self.team_b.get_lineup_for_minute(current_minute)
            
            # Obtenir PACE específic de cada quintet
            pace_a = getattr(lineup_a, 'pace', 100.0)
            pace_b = getattr(lineup_b, 'pace', 100.0)

            # MILLORA A: petit increment de pace per a l'equip local
            if self.home_team == 'a':
                pace_a *= self.home_pace_boost
            elif self.home_team == 'b':
                pace_b *= self.home_pace_boost

            pace_samples_a.append(pace_a)
            pace_samples_b.append(pace_b)

            # Calcular possessions que es generen aquest minut per cada equip
            # MILLORA E: soroll de pace reduït (std 0.20 → 0.10). La variància
            # alta no millorava l'encert del guanyador, només afegia soroll que
            # apropava els partits a una moneda a l'aire.
            variance_a = np.random.normal(1.0, 0.10)
            variance_b = np.random.normal(1.0, 0.10)
            variance_a = np.clip(variance_a, 0.80, 1.20)
            variance_b = np.clip(variance_b, 0.80, 1.20)
            
            possessions_generated_a = (pace_a / 48.0) * variance_a
            possessions_generated_b = (pace_b / 48.0) * variance_b
            
            pending_possessions_a += possessions_generated_a
            pending_possessions_b += possessions_generated_b
            
            # Simular possessions mentre n'hi hagi pendents (mínim 1 cada equip si n'hi ha)
            max_possessions_this_minute = 4  # Limitar per realisme
            possessions_simulated = 0
            
            while (pending_possessions_a >= 0.5 or pending_possessions_b >= 0.5) and possessions_simulated < max_possessions_this_minute:
                # Alternar entre equips, però donant prioritat a qui té més possessions pendents
                if pending_possessions_a > pending_possessions_b:
                    if pending_possessions_a >= 0.5:
                        # Possessió Team A
                        lineup_a.set_opposing_lineup(lineup_b)
                        path_a, points_a, stats_a = lineup_a.simulate_possession()
                        score_a += points_a
                        total_possessions_a += 1
                        quarter_poss_a += 1
                        pending_possessions_a -= 1.0
                        possessions_simulated += 1
                        
                        # Tracking de runs
                        if points_a > 0:
                            current_run_a += points_a
                            current_run_b = 0
                            if current_run_a > longest_run_a:
                                longest_run_a = current_run_a
                        elif points_a == 0:
                            current_run_a = 0
                        
                        # Lead changes
                        prev_diff = possession_log[-1]['score_a'] - possession_log[-1]['score_b'] if possession_log else 0
                        curr_diff = score_a - score_b
                        
                        if prev_diff != 0 and curr_diff != 0 and (prev_diff > 0) != (curr_diff > 0):
                            lead_changes += 1
                            if show_key_moments and not verbose:
                                print(f"   Q{quarter} {current_minute:.1f}' - 🔄 CANVI DE LIDERATGE → {score_a}-{score_b}")
                        
                        if curr_diff == 0 and prev_diff != 0:
                            ties += 1
                        
                        if curr_diff > biggest_lead_a:
                            biggest_lead_a = curr_diff
                        if curr_diff < 0 and abs(curr_diff) > biggest_lead_b:
                            biggest_lead_b = abs(curr_diff)
                        
                        if verbose and points_a > 0:
                            print(f"   Q{quarter} {current_minute:4.1f}' - {self.team_a.team_name}: +{points_a} → {score_a}-{score_b}")
                        
                        if show_key_moments and not verbose and current_run_a >= 8 and points_a > 0:
                            print(f"   Q{quarter} {current_minute:.1f}' - 🔥 {self.team_a.team_name} en RUN de {current_run_a} pts! → {score_a}-{score_b}")
                        
                        possession_log.append({
                            'possession': len(possession_log),
                            'quarter': quarter,
                            'minute': current_minute,
                            'team': 'A',
                            'score_a': score_a,
                            'score_b': score_b,
                            'points': points_a,
                            'pace_a': pace_a,
                            'pace_b': pace_b
                        })
                
                if pending_possessions_b >= 0.5:
                    # Possessió Team B
                    lineup_b.set_opposing_lineup(lineup_a)
                    path_b, points_b, stats_b = lineup_b.simulate_possession()
                    score_b += points_b
                    total_possessions_b += 1
                    quarter_poss_b += 1
                    pending_possessions_b -= 1.0
                    possessions_simulated += 1
                    
                    if points_b > 0:
                        current_run_b += points_b
                        current_run_a = 0
                        if current_run_b > longest_run_b:
                            longest_run_b = current_run_b
                    elif points_b == 0:
                        current_run_b = 0
                    
                    prev_diff = possession_log[-1]['score_a'] - possession_log[-1]['score_b'] if possession_log else 0
                    curr_diff = score_a - score_b
                    
                    if prev_diff != 0 and curr_diff != 0 and (prev_diff > 0) != (curr_diff > 0):
                        lead_changes += 1
                        if show_key_moments and not verbose:
                            print(f"   Q{quarter} {current_minute:.1f}' - 🔄 CANVI DE LIDERATGE → {score_a}-{score_b}")
                    
                    if curr_diff == 0 and prev_diff != 0:
                        ties += 1
                    
                    if curr_diff > biggest_lead_a:
                        biggest_lead_a = curr_diff
                    if curr_diff < 0 and abs(curr_diff) > biggest_lead_b:
                        biggest_lead_b = abs(curr_diff)
                    
                    if verbose and points_b > 0:
                        print(f"   Q{quarter} {current_minute:4.1f}' - {self.team_b.team_name}: +{points_b} → {score_a}-{score_b}")
                    
                    if show_key_moments and not verbose and current_run_b >= 8 and points_b > 0:
                        print(f"   Q{quarter} {current_minute:.1f}' - 🔥 {self.team_b.team_name} en RUN de {current_run_b} pts! → {score_a}-{score_b}")
                    
                    possession_log.append({
                        'possession': len(possession_log),
                        'quarter': quarter,
                        'minute': current_minute,
                        'team': 'B',
                        'score_a': score_a,
                        'score_b': score_b,
                        'points': points_b,
                        'pace_a': pace_a,
                        'pace_b': pace_b
                    })
            
            # Avançar minut
            current_minute += 1.0
            
            # Comprovar canvi de quarter
            if current_minute % QUARTER_MINUTES == 0 and current_minute < GAME_MINUTES:
                quarter_scores[quarter]['a'] = score_a - quarter_start_a
                quarter_scores[quarter]['b'] = score_b - quarter_start_b
                
                if show_quarters:
                    avg_pace_a = np.mean(pace_samples_a[-int(QUARTER_MINUTES):]) if pace_samples_a else 100.0
                    avg_pace_b = np.mean(pace_samples_b[-int(QUARTER_MINUTES):]) if pace_samples_b else 100.0
                    print(f"\n   Final Q{quarter}: {self.team_a.team_name} {quarter_scores[quarter]['a']} - {quarter_scores[quarter]['b']} {self.team_b.team_name}")
                    print(f"   Total: {score_a}-{score_b}")
                    print(f"   Possessions Q{quarter}: {self.team_a.team_name} {quarter_poss_a}, {self.team_b.team_name} {quarter_poss_b}")
                    print(f"   Pace Q{quarter}: {self.team_a.team_name} {avg_pace_a:.1f}, {self.team_b.team_name} {avg_pace_b:.1f}")
                
                quarter += 1
                quarter_start_a = score_a
                quarter_start_b = score_b
                quarter_poss_a = 0
                quarter_poss_b = 0
                
                if quarter <= 4 and show_quarters:
                    print(f"\n{'─'*70}")
                    print(f"📊 QUARTER {quarter}")
                    print(f"{'─'*70}")
        
        # Últim quarter
        if quarter == 4:
            quarter_scores[quarter]['a'] = score_a - quarter_start_a
            quarter_scores[quarter]['b'] = score_b - quarter_start_b
            
            if show_quarters:
                avg_pace_a = np.mean(pace_samples_a[-int(QUARTER_MINUTES):]) if pace_samples_a else 100.0
                avg_pace_b = np.mean(pace_samples_b[-int(QUARTER_MINUTES):]) if pace_samples_b else 100.0
                print(f"\n   Final Q{quarter}: {self.team_a.team_name} {quarter_scores[quarter]['a']} - {quarter_scores[quarter]['b']} {self.team_b.team_name}")
                print(f"   Total: {score_a}-{score_b}")
                print(f"   Possessions Q{quarter}: {self.team_a.team_name} {quarter_poss_a}, {self.team_b.team_name} {quarter_poss_b}")
                print(f"   Pace Q{quarter}: {self.team_a.team_name} {avg_pace_a:.1f}, {self.team_b.team_name} {avg_pace_b:.1f}")
        
        if verbose or show_key_moments:
            print(f"\n{'='*70}")
            print(f"🏆 FINAL: {self.team_a.team_name} {score_a} - {score_b} {self.team_b.team_name}")
            print(f"{'='*70}")
        
        # Pace mitjà de cada equip
        final_pace_a = np.mean(pace_samples_a) if pace_samples_a else 100.0
        final_pace_b = np.mean(pace_samples_b) if pace_samples_b else 100.0
        
        return {
            'score_a': score_a,
            'score_b': score_b,
            'winner': 'A' if score_a > score_b else ('B' if score_b > score_a else 'Tie'),
            'margin': abs(score_a - score_b),
            'team_a_name': self.team_a.team_name,
            'team_b_name': self.team_b.team_name,
            'quarter_scores': quarter_scores,
            'lead_changes': lead_changes,
            'ties': ties,
            'biggest_lead_a': biggest_lead_a,
            'biggest_lead_b': biggest_lead_b,
            'longest_run_a': longest_run_a,
            'longest_run_b': longest_run_b,
            'possessions_a': total_possessions_a,
            'possessions_b': total_possessions_b,
            'pace': (final_pace_a + final_pace_b) / 2,  # Pace combinat mitjà
            'pace_a': final_pace_a,  # NUEVO: Pace individual equip A
            'pace_b': final_pace_b,  # NUEVO: Pace individual equip B
            'ortg_a': (score_a / total_possessions_a * 100) if total_possessions_a > 0 else 0,
            'ortg_b': (score_b / total_possessions_b * 100) if total_possessions_b > 0 else 0,
            'possession_log': possession_log,
            'pace_samples_a': pace_samples_a,
            'pace_samples_b': pace_samples_b
        }
        """
        Simula un partit complet de 48 minuts NBA.
        El PACE es recalcula constantment segons els quintets actius.
        
        Args:
            verbose: Mostrar cada possessió
            show_quarters: Mostrar resum per quarters
            show_key_moments: Mostrar només moments clau (runs, lead changes)
            
        Returns:
            Diccionari amb resultats detallats
        """
        # Constants NBA
        GAME_MINUTES = 48.0
        QUARTER_MINUTES = 12.0
        
        # Inicialitzar
        score_a = 0
        score_b = 0
        possession_log = []
        quarter_scores = {1: {'a': 0, 'b': 0}, 2: {'a': 0, 'b': 0}, 
                         3: {'a': 0, 'b': 0}, 4: {'a': 0, 'b': 0}}
        
        # Stats avançades
        lead_changes = 0
        ties = 0
        biggest_lead_a = 0
        biggest_lead_b = 0
        current_run_a = 0
        current_run_b = 0
        longest_run_a = 0
        longest_run_b = 0
        
        total_possessions_a = 0
        total_possessions_b = 0
        
        # Tracking de pace per segment
        pace_samples = []
        
        if verbose or show_key_moments:
            print(f"\n{'='*70}")
            print(f"🏀 {self.team_a.team_name} vs {self.team_b.team_name}")
            print(f"{'='*70}\n")
        
        # NUEVO ENFOQUE: Simular minut a minut amb pace dinàmic
        current_minute = 0.0
        quarter = 1
        quarter_start_a = 0
        quarter_start_b = 0
        quarter_poss_a = 0
        quarter_poss_b = 0
        
        if show_quarters:
            print(f"\n{'─'*70}")
            print(f"📊 QUARTER {quarter}")
            print(f"{'─'*70}")
        
        while current_minute < GAME_MINUTES:
            # Obtenir quintets actius segons el minut actual
            lineup_a = self.team_a.get_lineup_for_minute(current_minute)
            lineup_b = self.team_b.get_lineup_for_minute(current_minute)
            
            # Obtenir PACE específic de cada quintet
            pace_a = getattr(lineup_a, 'pace', 100.0)
            pace_b = getattr(lineup_b, 'pace', 100.0)
            
            # Pace combinat (mitjana dels dos quintets actius)
            combined_pace = (pace_a + pace_b) / 2.0
            pace_samples.append(combined_pace)
            
            # Calcular possessions per aquest segment segons el pace
            # Pace = possessions per 48 min, així que per 1 min:
            possessions_per_minute = combined_pace / 48.0
            
            # VARIABILITAT NATURAL: El nombre de possessions per minut varia
            # Factors: fouls, timeouts, ritme de joc, etc.
            # Afegim variabilitat estocàstica del ±15% al voltant del pace
            # MILLORA E: variabilitat de pace reduïda (std 0.15 → 0.08)
            pace_variance = np.random.normal(1.0, 0.08)
            pace_variance = np.clip(pace_variance, 0.85, 1.15)  # Rang més estret
            
            adjusted_possessions_per_minute = possessions_per_minute * pace_variance
            
            # Simular possessions per aquest minut
            # Normalment ~2 possessions per minut (1 per equip)
            # Però ajustat pel pace actual i variabilitat
            num_possessions = max(1, int(round(adjusted_possessions_per_minute)))
            
            # Ocasionalment (5% del temps) hi ha possessions extra per fouls/timeouts
            if np.random.random() < 0.05:
                num_possessions += 1
            
            for _ in range(num_possessions):
                # Possessió Team A
                lineup_a.set_opposing_lineup(lineup_b)
                path_a, points_a, stats_a = lineup_a.simulate_possession()
                score_a += points_a
                total_possessions_a += 1
                quarter_poss_a += 1
                
                # Tracking de runs
                if points_a > 0:
                    current_run_a += points_a
                    current_run_b = 0
                    if current_run_a > longest_run_a:
                        longest_run_a = current_run_a
                elif points_a == 0:
                    current_run_a = 0
                
                # Possessió Team B
                lineup_b.set_opposing_lineup(lineup_a)
                path_b, points_b, stats_b = lineup_b.simulate_possession()
                score_b += points_b
                total_possessions_b += 1
                quarter_poss_b += 1
                
                # Tracking de runs
                if points_b > 0:
                    current_run_b += points_b
                    current_run_a = 0
                    if current_run_b > longest_run_b:
                        longest_run_b = current_run_b
                elif points_b == 0:
                    current_run_b = 0
                
                # Lead changes i ties
                prev_diff = possession_log[-1]['score_a'] - possession_log[-1]['score_b'] if possession_log else 0
                curr_diff = score_a - score_b
                
                if prev_diff != 0 and curr_diff != 0 and (prev_diff > 0) != (curr_diff > 0):
                    lead_changes += 1
                    if show_key_moments and not verbose:
                        print(f"   Q{quarter} {current_minute:.1f}' - 🔄 CANVI DE LIDERATGE → {score_a}-{score_b}")
                
                if curr_diff == 0 and prev_diff != 0:
                    ties += 1
                
                # Biggest leads
                if curr_diff > biggest_lead_a:
                    biggest_lead_a = curr_diff
                if curr_diff < 0 and abs(curr_diff) > biggest_lead_b:
                    biggest_lead_b = abs(curr_diff)
                
                # Mostrar possessions importants
                if verbose:
                    if points_a > 0 or points_b > 0:
                        if points_a > 0:
                            print(f"   Q{quarter} {current_minute:4.1f}' - {self.team_a.team_name}: +{points_a} → {score_a}-{score_b}")
                        if points_b > 0:
                            print(f"   Q{quarter} {current_minute:4.1f}' - {self.team_b.team_name}: +{points_b} → {score_a}-{score_b}")
                
                # Mostrar runs importants
                if show_key_moments and not verbose:
                    if current_run_a >= 8 and points_a > 0:
                        print(f"   Q{quarter} {current_minute:.1f}' - 🔥 {self.team_a.team_name} en RUN de {current_run_a} pts! → {score_a}-{score_b}")
                    if current_run_b >= 8 and points_b > 0:
                        print(f"   Q{quarter} {current_minute:.1f}' - 🔥 {self.team_b.team_name} en RUN de {current_run_b} pts! → {score_a}-{score_b}")
                
                # Log
                possession_log.append({
                    'possession': len(possession_log),
                    'quarter': quarter,
                    'minute': current_minute,
                    'score_a': score_a,
                    'score_b': score_b,
                    'points_a': points_a,
                    'points_b': points_b,
                    'pace': combined_pace,
                    'lineup_a_pace': pace_a,
                    'lineup_b_pace': pace_b
                })
            
            # Avançar minut
            current_minute += 1.0
            
            # Comprovar canvi de quarter
            if current_minute % QUARTER_MINUTES == 0 and current_minute < GAME_MINUTES:
                # Resum del quarter
                quarter_scores[quarter]['a'] = score_a - quarter_start_a
                quarter_scores[quarter]['b'] = score_b - quarter_start_b
                
                if show_quarters:
                    avg_pace_quarter = np.mean(pace_samples[-int(QUARTER_MINUTES):]) if pace_samples else 100.0
                    print(f"\n   Final Q{quarter}: {self.team_a.team_name} {quarter_scores[quarter]['a']} - {quarter_scores[quarter]['b']} {self.team_b.team_name}")
                    print(f"   Total: {score_a}-{score_b}")
                    print(f"   Possessions Q{quarter}: {self.team_a.team_name} {quarter_poss_a}, {self.team_b.team_name} {quarter_poss_b}")
                    print(f"   Pace mitjà Q{quarter}: {avg_pace_quarter:.1f}")
                
                # Preparar següent quarter
                quarter += 1
                quarter_start_a = score_a
                quarter_start_b = score_b
                quarter_poss_a = 0
                quarter_poss_b = 0
                
                if quarter <= 4 and show_quarters:
                    print(f"\n{'─'*70}")
                    print(f"📊 QUARTER {quarter}")
                    print(f"{'─'*70}")
        
        # Últim quarter
        if quarter == 4:
            quarter_scores[quarter]['a'] = score_a - quarter_start_a
            quarter_scores[quarter]['b'] = score_b - quarter_start_b
            
            if show_quarters:
                avg_pace_quarter = np.mean(pace_samples[-int(QUARTER_MINUTES):]) if pace_samples else 100.0
                print(f"\n   Final Q{quarter}: {self.team_a.team_name} {quarter_scores[quarter]['a']} - {quarter_scores[quarter]['b']} {self.team_b.team_name}")
                print(f"   Total: {score_a}-{score_b}")
                print(f"   Possessions Q{quarter}: {self.team_a.team_name} {quarter_poss_a}, {self.team_b.team_name} {quarter_poss_b}")
                print(f"   Pace mitjà Q{quarter}: {avg_pace_quarter:.1f}")
        
        # Resultats finals
        if verbose or show_key_moments:
            print(f"\n{'='*70}")
            print(f"🏆 FINAL: {self.team_a.team_name} {score_a} - {score_b} {self.team_b.team_name}")
            print(f"{'='*70}")
        
        # Calcular pace final (mitjana ponderada de tots els segments)
        final_pace = np.mean(pace_samples) if pace_samples else 100.0
        
        return {
            'score_a': score_a,
            'score_b': score_b,
            'winner': 'A' if score_a > score_b else ('B' if score_b > score_a else 'Tie'),
            'margin': abs(score_a - score_b),
            'team_a_name': self.team_a.team_name,
            'team_b_name': self.team_b.team_name,
            'quarter_scores': quarter_scores,
            'lead_changes': lead_changes,
            'ties': ties,
            'biggest_lead_a': biggest_lead_a,
            'biggest_lead_b': biggest_lead_b,
            'longest_run_a': longest_run_a,
            'longest_run_b': longest_run_b,
            'possessions_a': total_possessions_a,
            'possessions_b': total_possessions_b,
            'pace': final_pace,
            'ortg_a': (score_a / total_possessions_a * 100) if total_possessions_a > 0 else 0,
            'ortg_b': (score_b / total_possessions_b * 100) if total_possessions_b > 0 else 0,
            'possession_log': possession_log,
            'pace_samples': pace_samples  # Per anàlisi detallada
        }
    
    def print_game_summary(self, game_result: Dict):
        """Mostra resum detallat del partit."""
        
        print(f"\n{'='*70}")
        print(f"📊 RESUM DEL PARTIT")
        print(f"{'='*70}\n")
        
        # Box Score per quarters
        print("📋 BOX SCORE PER QUARTERS:\n")
        print(f"{'Equip':30s} {'Q1':>5s} {'Q2':>5s} {'Q3':>5s} {'Q4':>5s} {'FINAL':>6s}")
        print("-" * 70)
        
        q_scores = game_result['quarter_scores']
        team_a = game_result['team_a_name'][:28]
        team_b = game_result['team_b_name'][:28]
        
        print(f"{team_a:30s} {q_scores[1]['a']:5d} {q_scores[2]['a']:5d} {q_scores[3]['a']:5d} {q_scores[4]['a']:5d} {game_result['score_a']:6d}")
        print(f"{team_b:30s} {q_scores[1]['b']:5d} {q_scores[2]['b']:5d} {q_scores[3]['b']:5d} {q_scores[4]['b']:5d} {game_result['score_b']:6d}")
        
        # Stats avançades
        print(f"\n📈 ESTADÍSTIQUES AVANÇADES:\n")
        
        print(f"   Possessions:")
        print(f"      {team_a}: {game_result['possessions_a']}")
        print(f"      {team_b}: {game_result['possessions_b']}")
        print(f"      Pace: {game_result['pace']:.1f} poss/equip")
        
        print(f"\n   Offensive Rating (pts/100 poss):")
        print(f"      {team_a}: {game_result['ortg_a']:.1f}")
        print(f"      {team_b}: {game_result['ortg_b']:.1f}")
        
        print(f"\n   Dinàmica del partit:")
        print(f"      Canvis de lideratge: {game_result['lead_changes']}")
        print(f"      Empats: {game_result['ties']}")
        print(f"      Major avantatge {team_a}: +{game_result['biggest_lead_a']}")
        print(f"      Major avantatge {team_b}: +{game_result['biggest_lead_b']}")
        
        print(f"\n   Runs més llargs:")
        print(f"      {team_a}: {game_result['longest_run_a']} punts consecutius")
        print(f"      {team_b}: {game_result['longest_run_b']} punts consecutius")
        
        print(f"\n{'='*70}")


def load_team_rotation(team_name: str, 
                      lineups_csv: str = 'nba_lineups/nba_lineups.csv',
                      players_csv: str = 'nba_data_advanced/advanced_player_data.csv',
                      max_lineups: int = 20) -> RealisticLineupRotation:
    """
    Carrega la rotació d'un equip des del CSV (que té PACE).
    
    Args:
        team_name: Nom de l'equip (o part del nom)
        lineups_csv: Camí al CSV amb quintets (inclou PACE)
        players_csv: Camí al CSV amb jugadors
        max_lineups: Màxim de quintets a usar (per defecte 20, els més usats per minuts)
        
    Returns:
        RealisticLineupRotation object
    """
    import os
    
    if not os.path.exists(lineups_csv):
        raise FileNotFoundError(f"No s'ha trobat: {lineups_csv}\nExecuta: python nba_lineup_scraper.py")
    
    if not os.path.exists(players_csv):
        raise FileNotFoundError(f"No s'ha trobat: {players_csv}\nExecuta: python final_data_scraper.py")
    
    # Carregar CSV de quintets
    df_lineups = pd.read_csv(lineups_csv)
    player_df = pd.read_csv(players_csv)
    
    # Buscar equip
    team_matches = df_lineups[df_lineups['TEAM_NAME'].str.contains(team_name, case=False, na=False)]
    
    if len(team_matches) == 0:
        raise ValueError(f"Equip '{team_name}' no trobat al CSV")
    
    team_found = team_matches.iloc[0]['TEAM_NAME']
    
    # Filtrar quintets d'aquest equip i ordenar per minuts
    team_lineups = df_lineups[df_lineups['TEAM_NAME'] == team_found].copy()
    team_lineups = team_lineups.sort_values('MIN', ascending=False)
    
    # Convertir a format diccionari (compatible amb RealisticLineupRotation)
    lineups_data = []
    for _, row in team_lineups.iterrows():
        lineup_dict = {
            'lineup_id': row['GROUP_ID'],
            'players': row['PLAYER_NAMES'].split(' - ') if isinstance(row['PLAYER_NAMES'], str) else [],
            'minutes': row['MIN'],
            'games': row['GP'],
            'net_rating': row.get('PLUS_MINUS', 0) / row['GP'] if row['GP'] > 0 else 0,
            'pace': row['PACE'] if 'PACE' in row and pd.notna(row['PACE']) else 100.0,
            # Stats per calcular pace si no existeix
            'FGA': row.get('FGA', 0),
            'FTA': row.get('FTA', 0),
            'OREB': row.get('OREB', 0),
            'TOV': row.get('TOV', 0),
            'GP': row['GP'],
            'MIN': row['MIN'],
            'OFF_RATING': row.get('PTS', 0) / row['MIN'] * 100 if row['MIN'] > 0 else 100
        }
        lineups_data.append(lineup_dict)
    
    # Crear rotació
    rotation = RealisticLineupRotation(
        team_found,
        lineups_data,
        player_df,
        max_lineups=max_lineups
    )
    
    return rotation


# ============================================================================
# EXEMPLE D'ÚS
# ============================================================================

def main():
    """Exemple complet: simular Lakers vs Warriors amb quintets reals."""
    
    print("\n" + "="*70)
    print("🏀 SIMULACIÓ REALISTA DE PARTIT NBA")
    print("="*70)
    
    # 1. Carregar rotacions
    print("\n📊 1/3: Carregant quintets reals...\n")
    
    try:
        lakers_rotation = load_team_rotation("Lakers", max_lineups=20)
        warriors_rotation = load_team_rotation("Warriors", max_lineups=20)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nℹ️  Assegura't que has executat:")
        print("   1. python final_data_scraper.py")
        print("   2. python nba_lineup_scraper.py")
        return
    
    # 2. Mostrar plans de rotació (opcional)
    print("\n📋 2/3: Informació dels equips\n")
    print(f"✓ {lakers_rotation.team_name}: {len(lakers_rotation.markov_lineups)} quintets")
    print(f"✓ {warriors_rotation.team_name}: {len(warriors_rotation.markov_lineups)} quintets")
    
    # 3. Simular partit
    print("\n🎮 3/3: Simulant partit...\n")
    
    simulator = RealisticGameSimulator(lakers_rotation, warriors_rotation)
    
    # Simular amb moments clau (més net que verbose)
    game = simulator.simulate_game(
        verbose=False,           # No mostrar cada possessió
        show_quarters=True,      # Mostrar resums per quarter
        show_key_moments=True    # Mostrar runs i canvis de lideratge
    )
    
    # 4. Mostrar resum complet
    simulator.print_game_summary(game)
    
    print("\n" + "="*70)
    print("✅ SIMULACIÓ COMPLETADA!")
    print("="*70)
    print("\n💡 Aquest partit inclou:")
    print("  ✓ 4 quarters de 12 minuts")
    print("  ✓ ~100 possessions per equip (pace realista)")
    print("  ✓ Rotacions basades en minuts reals")
    print("  ✓ Tracking de runs, lead changes, i biggest leads")
    print("  ✓ Box score per quarters")
    print("  ✓ Offensive ratings")
    print()


if __name__ == "__main__":
    main()