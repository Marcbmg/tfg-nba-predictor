"""
Model Complet NBA amb Cadenes de Markov - Zones Detallades del Shot Chart
=========================================================================

Aquest model utilitza TOTES les zones del shot chart:
- Corner 3
- Above Break 3
- Mid-Range (8-16 ft)
- Long Mid-Range (16-22 ft)
- Paint (Non-RA)
- Layup (Restricted Area)
- Dunk

+ Estadístiques defensives que afecten els percentatges rivals
+ BLKA (Blocks Against) per calcular vulnerabilitat ofensiva REAL

VULNERABILITAT OFENSIVA (NOU!):
- Utilitza BLKA (Blocks Against) del CSV si està disponible
- BLKA indica quants taps rep cada jugador per partit
- Si BLKA no disponible, s'infereix per stats (REB, BLK, dunk_freq)
- Exemples:
  * Victor Wembanyama: BLKA=0.5 → vulnerability=0.5 (difícil tapar)
  * Dylan Harper: BLKA=0.8 → vulnerability=0.8 (moderadament fàcil)
  * Julian Reese: BLKA=1.8 → vulnerability=1.8 (molt fàcil tapar)

PROBABILITAT DE TAP:
P[Block] = Defensor_Block%_Zona × Atacant_Vulnerabilitat

Exemple: Dylan Harper vs Wembanyama (restricted area)
- Wembanyama defensive_restricted_block_pct = 28.5%
- Dylan Harper offensive_vulnerability = 0.8
- P[Layup → BLOCK] = 28.5% × 0.8 = 22.8%  ✅ REAL!

Estats del model (17 total):
OFENSIUS - ZONES DE TIR:
  0. Start - Inici de possessió
  1. Corner 3 - Tir de 3 des de cantonada
  2. Above Break 3 - Tir de 3 des de dalt de l'arc
  3. Mid-Range - Tir de 8-16 peus
  4. Long Mid-Range - Tir de 16-22 peus
  5. Paint - Tir a la zona pintada (non-RA)
  6. Layup - Entrada a cistella
  7. Dunk - Mate
  8. Turnover - Pèrdua de pilota
  9. Foul - Falta rebuda
  10. Free Throw - Tirs lliures

DEFENSIUS:
  11. Steal - Robatori
  12. Block - Tap
  13. Deflection - Desviament
  14. Offensive Rebound - Rebot ofensiu
  15. Defensive Rebound - Rebot defensiu

FINAL:
  16. End - Fi de possessió
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import sys


class GameStates:
    """Estats del joc amb zones detallades."""
    # Ofensius - Zones de tir
    START = 0
    CORNER_3 = 1
    ABOVE_BREAK_3 = 2
    MID_RANGE = 3
    LONG_MID_RANGE = 4
    PAINT = 5
    LAYUP = 6
    DUNK = 7
    TURNOVER = 8
    FOUL = 9
    FREE_THROW = 10
    
    # Defensius
    STEAL = 11
    BLOCK = 12
    DEFLECTION = 13
    OFFENSIVE_REBOUND = 14
    DEFENSIVE_REBOUND = 15
    
    # Final
    END = 16
    
    STATE_NAMES = [
        'Start',
        'Corner 3', 'Above Break 3',
        'Mid-Range (8-16ft)', 'Long Mid-Range (16-22ft)',
        'Paint', 'Layup', 'Dunk',
        'Turnover', 'Foul', 'Free Throw',
        'Steal', 'Block', 'Deflection',
        'Offensive Rebound', 'Defensive Rebound',
        'End'
    ]


class AdvancedPlayerFromData:
    """
    Jugador amb dades REALS del shot chart amb totes les zones.
    """
    
    def __init__(self, player_data: Dict):
        """
        Args:
            player_data: Diccionari amb totes les dades del jugador
        """
        self.player_id = player_data.get('player_id')
        self.name = player_data.get('player_name', 'Unknown')
        
        # Parsejar shot_chart_raw si existeix
        shot_chart_raw = player_data.get('shot_chart_raw')
        if isinstance(shot_chart_raw, str):
            try:
                shot_chart = json.loads(shot_chart_raw)
            except:
                shot_chart = {}
        else:
            shot_chart = shot_chart_raw if shot_chart_raw else {}
        
        # PERCENTATGES PER ZONA (des de shot chart)
        self.corner_3_pct = self._get_zone_pct(shot_chart, 'Corner 3', 0.36)
        self.above_break_3_pct = self._get_zone_pct(shot_chart, 'Above Break 3', 0.35)
        self.mid_range_pct = self._get_zone_pct(shot_chart, 'Mid-Range (8-16 ft)', 0.40)
        self.long_mid_range_pct = self._get_zone_pct(shot_chart, 'Long Mid-Range (16-22 ft)', 0.38)
        self.paint_pct = self._get_zone_pct(shot_chart, 'Paint (Non-RA)', 0.50)
        self.layup_pct = self._get_zone_pct(shot_chart, 'Layup', 0.65)
        if 'Layup' not in shot_chart and 'Restricted Area' in shot_chart:
            self.layup_pct = self._get_zone_pct(shot_chart, 'Restricted Area', 0.65)
        self.dunk_pct = self._get_zone_pct(shot_chart, 'Dunk', 0.92)
        self.ft_pct = 0.78
        
        # FREQÜÈNCIES PER ZONA
        self.corner_3_freq = self._get_zone_freq(shot_chart, 'Corner 3', 0.10)
        self.above_break_3_freq = self._get_zone_freq(shot_chart, 'Above Break 3', 0.20)
        self.mid_range_freq = self._get_zone_freq(shot_chart, 'Mid-Range (8-16 ft)', 0.15)
        self.long_mid_range_freq = self._get_zone_freq(shot_chart, 'Long Mid-Range (16-22 ft)', 0.10)
        self.paint_freq = self._get_zone_freq(shot_chart, 'Paint (Non-RA)', 0.10)
        self.layup_freq = self._get_zone_freq(shot_chart, 'Layup', 0.20)
        if 'Layup' not in shot_chart and 'Restricted Area' in shot_chart:
            self.layup_freq = self._get_zone_freq(shot_chart, 'Restricted Area', 0.20)
        self.dunk_freq = self._get_zone_freq(shot_chart, 'Dunk', 0.10)
        
        # Normalitzar freqüències
        total_freq = (self.corner_3_freq + self.above_break_3_freq + 
                     self.mid_range_freq + self.long_mid_range_freq +
                     self.paint_freq + self.layup_freq + self.dunk_freq)
        
        if total_freq > 0:
            self.corner_3_freq /= total_freq
            self.above_break_3_freq /= total_freq
            self.mid_range_freq /= total_freq
            self.long_mid_range_freq /= total_freq
            self.paint_freq /= total_freq
            self.layup_freq /= total_freq
            self.dunk_freq /= total_freq
        
        # ESTADÍSTIQUES DEFENSIVES - PERCENTATGES DELS RIVALS
        self.def_fg_pct_against = player_data.get('def_fg_pct_against', 0.47)
        self.def_2pt_pct_against = player_data.get('def_2pt_pct_against', 0.49)
        self.def_3pt_pct_against = player_data.get('def_3pt_pct_against', 0.36)
        self.def_fga_against = player_data.get('def_fga_against', 0)
        
        # HUSTLE STATS
        self.deflections = player_data.get('deflections', 0)
        self.contested_shots = player_data.get('contested_shots', 0)
        
        # Calcular taxes
        total_possessions = 100 * 70
        self.steal_rate = 0.015
        
        # BLOCK RATE segons fórmula Basketball-Reference (general)
        self.block_rate = self._calculate_block_rate(player_data)
        
        # BLOCK RATES PER ZONA (Solució 2: FGA Weighted)
        # Aquests vénen directament del CSV calculats pel scraper
        # IMPORTANT: Per matriu individual, usar MITJANA NBA com a defensa base
        # La defensa real es calcularà quan s'assigni l'opposing lineup
        
        # Mitjana NBA de block% per zona (defensa típica)
        NBA_AVG_RESTRICTED_BLOCK = 8.0   # 8% layups tapats (mitjana NBA)
        NBA_AVG_PAINT_BLOCK = 6.0        # 6% paint tapats
        NBA_AVG_MIDRANGE_BLOCK = 3.0     # 3% mid-range tapats
        NBA_AVG_LONG_MID_BLOCK = 2.0     # 2% long mid tapats
        NBA_AVG_3PT_BLOCK = 0.5          # 0.5% triples tapats
        
        # Block% DEFENSIU del jugador (quan ELL defensa)
        # Aquests vénen directament del CSV i són CORRECTES
        self.defensive_restricted_block_pct = player_data.get('restricted_block_pct', NBA_AVG_RESTRICTED_BLOCK)
        self.defensive_paint_block_pct = player_data.get('paint_block_pct', NBA_AVG_PAINT_BLOCK)
        self.defensive_midrange_block_pct = player_data.get('midrange_block_pct', NBA_AVG_MIDRANGE_BLOCK)
        self.defensive_long_midrange_block_pct = player_data.get('long_midrange_block_pct', NBA_AVG_LONG_MID_BLOCK)
        self.defensive_above_break_3_block_pct = player_data.get('above_break_3_block_pct', NBA_AVG_3PT_BLOCK)
        self.defensive_corner_3_block_pct = player_data.get('corner_3_block_pct', NBA_AVG_3PT_BLOCK)
        
        # VULNERABILITAT OFENSIVA: Bases petits són més fàcils de tapar que pivots grans
        self.offensive_vulnerability = self._calculate_offensive_vulnerability(player_data)
        
        # Block% ofensiu per matriu INDIVIDUAL (mitjana NBA × vulnerabilitat)
        # Aquests s'usen només per la matriu individual sense rival
        self.restricted_block_pct = NBA_AVG_RESTRICTED_BLOCK * self.offensive_vulnerability
        self.paint_block_pct = NBA_AVG_PAINT_BLOCK * self.offensive_vulnerability
        self.midrange_block_pct = NBA_AVG_MIDRANGE_BLOCK * self.offensive_vulnerability
        self.long_midrange_block_pct = NBA_AVG_LONG_MID_BLOCK * self.offensive_vulnerability
        self.above_break_3_block_pct = NBA_AVG_3PT_BLOCK * self.offensive_vulnerability
        self.corner_3_block_pct = NBA_AVG_3PT_BLOCK * self.offensive_vulnerability
        
        self.deflection_rate = self.deflections / total_possessions if total_possessions > 0 else 0.04
        
        self.turnover_rate = 0.12
        self.offensive_reb_rate = 0.05
        self.defensive_reb_rate = 0.12
        self.foul_draw_rate = 0.08
    
    def _calculate_block_rate(self, player_data: Dict) -> float:
        """
        Calcula el block rate segons la fórmula oficial de Basketball-Reference.
        
        Fórmula:
        BLK% = 100 * (BLK * (Tm_MP / 5)) / (MP * (Opp_FGA - Opp_3PA))
        
        On:
        - BLK: Taps totals del jugador
        - Tm_MP: Minuts totals de l'equip (240 sense pròrroga)
        - MP: Minuts jugats pel jugador
        - Opp_FGA: Tirs de camp intentats pels rivals (ara del CSV!)
        - Opp_3PA: Tirs de 3 punts intentats pels rivals (ara del CSV!)
        
        Aquesta mètrica estima el % de tirs de 2 punts rivals que el jugador tapa.
        
        Returns:
            Block rate com a probabilitat (0.0 a 1.0)
        """
        # Extreure dades del jugador
        blk = player_data.get('BLK', 0)  # Taps per partit
        mp = player_data.get('MPG', player_data.get('MP', 30))  # Minuts per partit
        
        # Dades de l'equip/rivals
        # IMPORTANT: Ara això hauria de venir del CSV amb dades reals!
        tm_mp = player_data.get('team_mp', 240)  # Minuts totals equip
        opp_fga = player_data.get('opp_fga', player_data.get('OPP_FGA', 85))  # FGA rivals (del CSV!)
        opp_3pa = player_data.get('opp_3pa', player_data.get('OPP_3PA', 35))  # 3PA rivals (del CSV!)
        
        # Aplicar fórmula Basketball-Reference
        if mp > 0 and (opp_fga - opp_3pa) > 0:
            # BLK% = 100 * (BLK * (Tm_MP / 5)) / (MP * (Opp_FGA - Opp_3PA))
            blk_percentage = 100 * (blk * (tm_mp / 5)) / (mp * (opp_fga - opp_3pa))
        else:
            # Si no hi ha dades suficients, usar valor per defecte baix
            blk_percentage = 1.5  # 1.5% per defecte
        
        # Convertir percentatge a probabilitat (0-1)
        block_rate = blk_percentage / 100
        
        # Aplicar límits raonables
        # Mínim: 0.1% (fins i tot jugadors sense taps tenen una mínima probabilitat)
        # Màxim: 15% (el millor tapador de la història no arriba a més)
        block_rate = max(0.001, min(block_rate, 0.15))
        
        return block_rate
    
    def _calculate_offensive_vulnerability(self, player_data: Dict) -> float:
        """
        Calcula la vulnerabilitat ofensiva del jugador a ser tapat.
        
        PRIORITAT 1: Usar BLKA (Blocks Against) si està disponible al CSV
        ----------------------------------------------------------------
        BLKA = Taps rebuts per partit (dada REAL de NBA.com)
        - Wembanyama: BLKA = 0.5 → vulnerability = 0.5 (molt difícil tapar)
        - Dylan Harper: BLKA = 0.8 → vulnerability = 0.8 
        - Stephen Curry: BLKA = 0.6 → vulnerability = 0.6
        - Julian Reese: BLKA = 1.8 → vulnerability = 1.8 (molt fàcil tapar)
        
        PRIORITAT 2: Inferir per stats si BLKA no disponible
        -----------------------------------------------------
        Combina 3 factors:
        1. Alçada/Posició (REB, BLK)
        2. Atletisme (dunk_freq)
        3. Estil de joc (layup_freq vs dunk_freq)
        
        Returns:
            Multiplicador de vulnerabilitat:
            - 0.3-0.7 = Molt difícil de tapar (pivots, atletes elite)
            - 0.8-1.2 = Normal (mitjana NBA = 1.0)
            - 1.3-2.5 = Fàcil de tapar (bases petits, tiradors)
        
        Ús al model:
            P[Block] = Defensor_Block%_Zona × Atacant_Vulnerabilitat
            
            Exemple: Dylan Harper layup vs Wembanyama
            - Wembanyama defensive_restricted_block% = 28.5%
            - Dylan Harper vulnerability = 0.8 (BLKA real)
            - P[Layup → BLOCK] = 28.5% × 0.8 = 22.8%  ✅
        """
        # ===================================
        # OPCIÓ 1: Usar BLKA REAL (MILLOR!)
        # ===================================
        blka = player_data.get('BLKA', None)
        fga = player_data.get('FGA', 0)
        
        if blka is not None and blka > 0:
            # BLKA = Blocks Against (taps rebuts per partit)
            # Font: NBA.com Official Stats o estimació del scraper
            
            # Mitjana NBA: ~1.0 BLKA per partit
            NBA_AVG_BLKA = 1.0
            
            # Vulnerabilitat = BLKA del jugador / mitjana NBA
            # Exemples:
            # - Wembanyama BLKA=0.5 → vulnerability=0.5 (difícil)
            # - Harper BLKA=0.8 → vulnerability=0.8
            # - Reese BLKA=1.8 → vulnerability=1.8 (fàcil)
            vulnerability = blka / NBA_AVG_BLKA
            
            # Límits raonables (0.3 a 2.5)
            # 0.3 = Quasi impossible de tapar (Shaq, Wembanyama elite)
            # 2.5 = Extremadament fàcil de tapar (bases petits vulnerables)
            vulnerability = max(0.3, min(vulnerability, 2.5))
            
            return vulnerability
        
        # ===================================
        # OPCIÓ 2: Inferir per stats
        # ===================================
        # Només s'usa si BLKA no està disponible
        # Aquest mètode és menys precís però millor que res
        # Extreure stats
        reb = player_data.get('REB', 0)
        ast = player_data.get('AST', 0)
        blk = player_data.get('BLK', 0)
        dunk_freq = player_data.get('dunk_freq', 0)
        layup_freq = player_data.get('layup_freq', 0)
        
        # ===================================
        # FACTOR 1: ALÇADA/POSICIÓ
        # ===================================
        # Pivots grans són molt difícils de tapar
        if reb > 10 or blk > 2.0:
            # Pivot elite (Wembanyama 11.1 REB, 3.6 BLK)
            height_factor = 0.5  # Molt difícil de tapar!
        elif reb > 8 or blk > 1.0:
            # Pivot normal (Jokić, Gobert)
            height_factor = 0.7
        elif reb > 6:
            # Aler-pivot (AD, LeBron)
            height_factor = 0.85
        elif reb > 4:
            # Aler/Escorta
            height_factor = 1.0  # Normal
        else:
            # Base (Curry 4.5 REB, Dylan Harper probablement <4 REB)
            height_factor = 1.3  # Més fàcil de tapar
        
        # ===================================
        # FACTOR 2: ATLETISME
        # ===================================
        # Jugadors atlètics (Ja Morant, Zion) són més difícils de tapar
        if dunk_freq > 0.20:
            # Molt atlètic
            athleticism_factor = 0.75
        elif dunk_freq > 0.10:
            # Atlètic
            athleticism_factor = 0.90
        elif dunk_freq > 0.05:
            # Normal
            athleticism_factor = 1.0
        else:
            # Poc atlètic (bases tiradors)
            athleticism_factor = 1.15
        
        # ===================================
        # FACTOR 3: ESTIL DE JOC
        # ===================================
        # Floaters vs dunks
        if layup_freq > 0.25 and dunk_freq < 0.05:
            # Bases que fan molts floaters (Tony Parker, Curry)
            # Més fàcils de tapar
            style_factor = 1.2
        elif dunk_freq > 0.15:
            # Jugadors que esmaixen molt (Giannis, Zion)
            # Difícils de tapar
            style_factor = 0.85
        else:
            # Normal
            style_factor = 1.0
        
        # ===================================
        # COMBINAR FACTORS
        # ===================================
        vulnerability = height_factor * athleticism_factor * style_factor
        
        # Límits raonables (0.5 a 1.8)
        vulnerability = max(0.5, min(vulnerability, 1.8))
        
        return vulnerability
    
    def _get_zone_pct(self, shot_chart: Dict, zone_name: str, default: float) -> float:
        """Obté percentatge d'una zona del shot chart."""
        if zone_name in shot_chart:
            pct = shot_chart[zone_name].get('fg_pct', default)
            return pct if pct > 0 else default
        return default
    
    def _get_zone_freq(self, shot_chart: Dict, zone_name: str, default: float) -> float:
        """Obté freqüència d'una zona del shot chart."""
        if zone_name in shot_chart:
            freq = shot_chart[zone_name].get('frequency', default)
            return freq if freq > 0 else default
        return default
    
    def __repr__(self):
        return f"AdvancedPlayer({self.name})"
    
    def _build_transition_matrix(self, opposing_defense_impact: float = 1.0) -> np.ndarray:
        """
        Construeix la matriu de transició 17x17 INDIVIDUAL del jugador.

        Aquesta versió deriva matemàticament les taxes de falta i rebot ofensiu
        per zona a partir de les dades individuals del jugador (foul_draw_rate,
        offensive_reb_rate, freqüències per zona), garantint que la mitjana
        ponderada coincideixi amb la taxa global del CSV.

        FÓRMULA RIGOROSA per cada zona z:
            rate(z) = w(z) × k_jugador
        on:
            w(z)        = pes estructural de la zona (basat en física del joc)
            k_jugador   = global_rate / Σ(freq(z) × w(z))   [calibració per jugador]

        Args:
            opposing_defense_impact: Factor d'impacte defensiu del rival

        Returns:
            Matriu numpy 17x17 amb probabilitats de transició
        """
        n = 17
        P = np.zeros((n, n))

        # ══════════════════════════════════════════════════════════════════
        # PESOS ESTRUCTURALS PER ZONA (basats en física del joc)
        # ══════════════════════════════════════════════════════════════════
        ZONE_FOUL_WEIGHTS = {
            'corner_3':       0.07,   # Defensors lluny, poc contacte
            'above_break_3':  0.09,   # Defensors lluny
            'mid_range':      0.15,   # Contacte moderat
            'long_mid_range': 0.13,
            'paint':          0.34,   # Contacte alt
            'layup':          0.66,   # Contacte molt alt
            'dunk':           0.98,   # AND-1 freqüents
        }
        ZONE_OREB_WEIGHTS = {
            'corner_3':       0.65,   # Rebots molt llargs
            'above_break_3':  0.80,
            'mid_range':      0.85,
            'long_mid_range': 0.78,
            'paint':          1.05,   # Rebots prop de la cistella
            'layup':          1.20,
            'dunk':           1.10,
        }

        # ══════════════════════════════════════════════════════════════════
        # CÀLCUL DE TAXES PER ZONA DERIVADES DE LES DADES DEL JUGADOR
        # ══════════════════════════════════════════════════════════════════
        frequencies = {
            'corner_3':       self.corner_3_freq,
            'above_break_3':  self.above_break_3_freq,
            'mid_range':      self.mid_range_freq,
            'long_mid_range': self.long_mid_range_freq,
            'paint':          self.paint_freq,
            'layup':          self.layup_freq,
            'dunk':           self.dunk_freq,
        }

        # Taxes de falta per zona (derivades de self.foul_draw_rate)
        foul_weighted_avg = sum(frequencies[z] * ZONE_FOUL_WEIGHTS[z]
                                for z in ZONE_FOUL_WEIGHTS)
        if foul_weighted_avg > 0 and self.foul_draw_rate > 0:
            k_foul = self.foul_draw_rate / foul_weighted_avg
            foul_rates = {z: min(0.95, max(0.005, ZONE_FOUL_WEIGHTS[z] * k_foul))
                          for z in ZONE_FOUL_WEIGHTS}
        else:
            foul_rates = {z: 0.02 for z in ZONE_FOUL_WEIGHTS}

        # Taxes de rebot ofensiu per zona (derivades de self.offensive_reb_rate)
        # Si el valor del CSV està mal escalat, reescalar-lo perquè sigui
        # consistent amb les taxes NBA reals (~25% mitjana de tirs fallats)
        oreb_global = self.offensive_reb_rate
        if oreb_global < 0.05:
            # CSV en escala diferent; reescalar per fer-lo consistent
            oreb_global = min(0.5, oreb_global * 50)

        oreb_weighted_avg = sum(frequencies[z] * ZONE_OREB_WEIGHTS[z]
                                for z in ZONE_OREB_WEIGHTS)
        if oreb_weighted_avg > 0 and oreb_global > 0:
            k_oreb = oreb_global / oreb_weighted_avg
            oreb_rates = {z: min(0.50, max(0.05, ZONE_OREB_WEIGHTS[z] * k_oreb))
                          for z in ZONE_OREB_WEIGHTS}
        else:
            oreb_rates = {z: 0.25 for z in ZONE_OREB_WEIGHTS}

        # ══════════════════════════════════════════════════════════════════
        # START (0) - Selecció de zona de tir
        # ══════════════════════════════════════════════════════════════════
        total_threat = (self.steal_rate + self.deflection_rate) * 1.2
        safe_pct = max(0.75, 1.0 - self.turnover_rate - total_threat)

        P[GameStates.START, GameStates.CORNER_3] = self.corner_3_freq * safe_pct
        P[GameStates.START, GameStates.ABOVE_BREAK_3] = self.above_break_3_freq * safe_pct
        P[GameStates.START, GameStates.MID_RANGE] = self.mid_range_freq * safe_pct
        P[GameStates.START, GameStates.LONG_MID_RANGE] = self.long_mid_range_freq * safe_pct
        P[GameStates.START, GameStates.PAINT] = self.paint_freq * safe_pct
        P[GameStates.START, GameStates.LAYUP] = self.layup_freq * safe_pct
        P[GameStates.START, GameStates.DUNK] = self.dunk_freq * safe_pct

        P[GameStates.START, GameStates.TURNOVER] = self.turnover_rate
        P[GameStates.START, GameStates.STEAL] = self.steal_rate
        P[GameStates.START, GameStates.DEFLECTION] = self.deflection_rate

        # ══════════════════════════════════════════════════════════════════
        # FUNCIÓ AUXILIAR PER CONFIGURAR UNA ZONA
        # ══════════════════════════════════════════════════════════════════
        def configure_zone(zone_state, zone_name, make_pct, block_pct_value,
                           apply_def_impact=True):
            """Configura una zona usant les taxes derivades del propi jugador."""
            # 1. Anotar (ajustat per defensa rival)
            if apply_def_impact:
                make_rate = min(0.95, make_pct * opposing_defense_impact)
            else:
                make_rate = min(0.95, make_pct)
            miss_rate = 1.0 - make_rate

            # 2. Tap (del CSV defensiu)
            block_prob = min(0.30, block_pct_value / 100)

            # 3. Falta (DERIVADA RIGOROSAMENT del foul_draw_rate del jugador)
            foul_prob = foul_rates[zone_name]

            # 4. Garantir consistència: block + foul <= miss_rate
            events_no_make = block_prob + foul_prob
            if events_no_make > miss_rate * 0.95:
                scale = (miss_rate * 0.95) / events_no_make
                block_prob *= scale
                foul_prob *= scale

            # 5. La resta es reparteix entre OREB i DREB usant taxa derivada
            remaining = max(0, miss_rate - block_prob - foul_prob)
            oreb_prob = oreb_rates[zone_name]

            # 6. Assignar
            P[zone_state, GameStates.END] = make_rate
            P[zone_state, GameStates.BLOCK] = block_prob
            P[zone_state, GameStates.FOUL] = foul_prob
            P[zone_state, GameStates.OFFENSIVE_REBOUND] = remaining * oreb_prob
            P[zone_state, GameStates.DEFENSIVE_REBOUND] = remaining * (1 - oreb_prob)

        # ══════════════════════════════════════════════════════════════════
        # CONFIGURAR TOTES LES ZONES (una única crida per zona)
        # ══════════════════════════════════════════════════════════════════
        configure_zone(GameStates.CORNER_3, 'corner_3',
                       self.corner_3_pct, self.corner_3_block_pct)
        configure_zone(GameStates.ABOVE_BREAK_3, 'above_break_3',
                       self.above_break_3_pct, self.above_break_3_block_pct)
        configure_zone(GameStates.MID_RANGE, 'mid_range',
                       self.mid_range_pct, self.midrange_block_pct)
        configure_zone(GameStates.LONG_MID_RANGE, 'long_mid_range',
                       self.long_mid_range_pct, self.long_midrange_block_pct)
        configure_zone(GameStates.PAINT, 'paint',
                       self.paint_pct, self.paint_block_pct)
        configure_zone(GameStates.LAYUP, 'layup',
                       self.layup_pct, self.restricted_block_pct)
        configure_zone(GameStates.DUNK, 'dunk',
                       self.dunk_pct, self.restricted_block_pct * 0.5)

        # ══════════════════════════════════════════════════════════════════
        # ESTATS FINALS I DEFENSIUS
        # ══════════════════════════════════════════════════════════════════
        P[GameStates.TURNOVER, GameStates.END] = 1.0
        P[GameStates.FOUL, GameStates.FREE_THROW] = 1.0
        P[GameStates.FREE_THROW, GameStates.END] = 1.0
        P[GameStates.STEAL, GameStates.END] = 1.0
        P[GameStates.BLOCK, GameStates.OFFENSIVE_REBOUND] = 0.3
        P[GameStates.BLOCK, GameStates.DEFENSIVE_REBOUND] = 0.7
        P[GameStates.DEFLECTION, GameStates.TURNOVER] = 0.6
        P[GameStates.DEFLECTION, GameStates.START] = 0.4
        P[GameStates.OFFENSIVE_REBOUND, GameStates.START] = 1.0
        P[GameStates.DEFENSIVE_REBOUND, GameStates.END] = 1.0
        P[GameStates.END, GameStates.END] = 1.0

        # Normalitzar cada fila
        for i in range(n):
            row_sum = P[i].sum()
            if row_sum > 0:
                P[i] /= row_sum
            else:
                P[i, GameStates.END] = 1.0

        return P


class AdvancedLineupFromData:
    """
    Quintet amb matriu de transició que usa totes les zones i defensive stats.
    """
    
    def __init__(self, name: str, players: List[AdvancedPlayerFromData],
                 opposing_lineup: Optional['AdvancedLineupFromData'] = None,
                 lineups_df: Optional[pd.DataFrame] = None):
        """
        Args:
            name: Nom del quintet
            players: 5 jugadors amb dades reals
            opposing_lineup: Quintet rival (per usar les seves defensive stats)
            lineups_df: DataFrame de nba_lineups.csv per calcular factor
                       bayesià de calibració basat en Plus/Minus del quintet
        """
        assert len(players) == 5, "Cal exactament 5 jugadors"

        self.name = name
        self.players = players
        self.opposing_lineup = opposing_lineup
        self.lineups_df = lineups_df

        # MILLORA A: multiplicador d'avantatge de pista local.
        # 1.0 = neutre. El simulador el posa >1.0 a l'equip local perquè
        # tregui una mica més d'eficiència ofensiva (és l'origen principal
        # del home-court advantage real de la NBA).
        self.home_court_factor = 1.0

        # Calcular estadístiques mitjanes
        self._calculate_team_averages()

        # NOU: Calcular factor de calibració bayesià del quintet
        # (basat en Plus/Minus i minuts jugats junts)
        self._compute_lineup_calibration()

        # Calcular defensive impact del rival
        self._calculate_opposing_defense()

        # Construir matriu
        self.matrix = self._build_transition_matrix()

        # Crear cadena de Markov
        self.markov_chain = AdvancedMarkovChain(self.matrix)

    def set_opposing_lineup(self, opposing_lineup: 'AdvancedLineupFromData'):
        """Estableix el quintet rival i recalcula la matriu."""
        self.opposing_lineup = opposing_lineup
        self._calculate_opposing_defense()
        self.matrix = self._build_transition_matrix()
        self.markov_chain = AdvancedMarkovChain(self.matrix)
    
    def _calculate_team_averages(self):
        """Calcula mitjanes del quintet per cada zona."""
        # Percentatges per zona
        self.avg_corner_3_pct = np.mean([p.corner_3_pct for p in self.players])
        self.avg_above_break_3_pct = np.mean([p.above_break_3_pct for p in self.players])
        self.avg_mid_range_pct = np.mean([p.mid_range_pct for p in self.players])
        self.avg_long_mid_range_pct = np.mean([p.long_mid_range_pct for p in self.players])
        self.avg_paint_pct = np.mean([p.paint_pct for p in self.players])
        self.avg_layup_pct = np.mean([p.layup_pct for p in self.players])
        self.avg_dunk_pct = np.mean([p.dunk_pct for p in self.players])
        self.avg_ft_pct = np.mean([p.ft_pct for p in self.players])
        
        # Freqüències per zona
        self.avg_corner_3_freq = np.mean([p.corner_3_freq for p in self.players])
        self.avg_above_break_3_freq = np.mean([p.above_break_3_freq for p in self.players])
        self.avg_mid_range_freq = np.mean([p.mid_range_freq for p in self.players])
        self.avg_long_mid_range_freq = np.mean([p.long_mid_range_freq for p in self.players])
        self.avg_paint_freq = np.mean([p.paint_freq for p in self.players])
        self.avg_layup_freq = np.mean([p.layup_freq for p in self.players])
        self.avg_dunk_freq = np.mean([p.dunk_freq for p in self.players])
        
        # Defensa pròpia (com defensen)
        self.avg_def_fg_pct_against = np.mean([p.def_fg_pct_against for p in self.players])
        self.avg_def_2pt_pct_against = np.mean([p.def_2pt_pct_against for p in self.players])
        self.avg_def_3pt_pct_against = np.mean([p.def_3pt_pct_against for p in self.players])
        
        # Altres
        self.avg_steal_rate = np.mean([p.steal_rate for p in self.players])
        self.avg_block_rate = np.mean([p.block_rate for p in self.players])
        
        # Block % DEFENSIU per zona (quan aquest quintet DEFENSA)
        # Aquests són els valors que s'usaran quan l'OPONENT ataqui
        self.avg_defensive_restricted_block_pct = np.mean([p.defensive_restricted_block_pct for p in self.players])
        self.avg_defensive_paint_block_pct = np.mean([p.defensive_paint_block_pct for p in self.players])
        self.avg_defensive_midrange_block_pct = np.mean([p.defensive_midrange_block_pct for p in self.players])
        self.avg_defensive_long_midrange_block_pct = np.mean([p.defensive_long_midrange_block_pct for p in self.players])
        self.avg_defensive_above_break_3_block_pct = np.mean([p.defensive_above_break_3_block_pct for p in self.players])
        self.avg_defensive_corner_3_block_pct = np.mean([p.defensive_corner_3_block_pct for p in self.players])
        
        # VULNERABILITAT OFENSIVA (quan aquest quintet ATACA)
        # Mitjana de vulnerabilitat dels 5 jugadors
        self.avg_offensive_vulnerability = np.mean([p.offensive_vulnerability for p in self.players])
        self.avg_deflection_rate = np.mean([p.deflection_rate for p in self.players])
        self.avg_turnover_rate = np.mean([p.turnover_rate for p in self.players])
        self.avg_offensive_reb_rate = np.mean([p.offensive_reb_rate for p in self.players])
        self.avg_defensive_reb_rate = np.mean([p.defensive_reb_rate for p in self.players])
        self.avg_foul_draw_rate = np.mean([p.foul_draw_rate for p in self.players])

    def _compute_lineup_calibration(self):
        """
        Calcula el factor de calibració ofensiva del quintet aplicant un
        estimador Bayesià de tipus shrinkage (Efron & Morris, 1977).

        IDEA:
        Combinem dues fonts d'informació per estimar l'eficiència del quintet:
          1. Dades INDIVIDUALS dels 5 jugadors (estimació prior)
          2. Plus/Minus del quintet quan ha jugat junt (evidència empírica)

        FÓRMULA:
            of_factor = α + (1 - α) · (ORtg_quintet / ORtg_NBA)
            α = n_0 / (n_0 + minuts_jugats)

        On α regula la confiança en les dades empíriques del quintet:
          - Si el quintet ha jugat MOLTS minuts → α petit → confiar en Plus/Minus
          - Si el quintet ha jugat POCS minuts → α gran → confiar en individual
          - Si el quintet NO ha jugat mai → α = 1 → només dades individuals

        Aquest factor captura SINERGIES entre jugadors que les estadístiques
        individuals no poden veure (química, sistemes tàctics, coaching).
        """
        # Paràmetres del model bayesià
        # MILLORA B: N_0 baixat de 10 → 4. Amb N_0=10 gairebé tots els quintets
        # (que tenen pocs minuts junts) quedaven arrossegats cap al prior i la
        # força real s'aixafava. Amb N_0=4 les dades empíriques del quintet
        # tenen molt més pes quan existeixen.
        N_0 = 4.0
        NBA_AVG_OF_RATING = 115.0  # ORtg mitjà NBA temporada actual

        # MILLORA B: rangs de clip. El prior d'equip i l'of_factor final es
        # mantenen acotats per estabilitat, però amb prou amplada perquè els
        # equips bons i dolents siguin REALMENT diferents (abans tot ~[0.97,1.03]).
        PRIOR_CLIP = (0.85, 1.15)
        OF_CLIP = (0.80, 1.20)

        # Valors per defecte
        self.of_factor = 1.0
        self.lineup_minutes = 0.0
        self.alpha_bayes = 1.0
        self.lineup_found = False

        # Si no tenim dades de lineups, només dades individuals
        if self.lineups_df is None or len(self.lineups_df) == 0:
            return

        # Generar GROUP_ID del nostre quintet
        # Format del CSV: "-203497-203944-1628978-1630162-1630183-"
        # (player_ids ordenats numèricament, separats per '-' amb '-' inicial i final)
        try:
            player_ids = sorted([int(p.player_id) for p in self.players])
        except (AttributeError, ValueError, TypeError):
            # Si els jugadors no tenen player_id, no podem buscar
            return

        # ──────────────────────────────────────────────────────────────────
        # MILLORA B: PRIOR ANCORAT A LA FORÇA REAL DE L'EQUIP
        # En comptes de fer shrinkage cap a 1.0 (mitjana de la lliga, que
        # converteix tots els equips en "mitjans"), fem shrinkage cap a
        # l'eficiència ofensiva REAL de l'equip al qual pertany el quintet.
        # Així, fins i tot un quintet de banqueta sense dades pròpies hereta
        # la identitat ofensiva del seu equip i no es torna genèric.
        # ──────────────────────────────────────────────────────────────────
        team_prior = self._estimate_team_off_factor(
            player_ids, NBA_AVG_OF_RATING, PRIOR_CLIP
        )

        # Construir GROUP_ID com al CSV i buscar el quintet exacte
        group_id_target = "-" + "-".join(str(pid) for pid in player_ids) + "-"
        matches = self.lineups_df[self.lineups_df['GROUP_ID'] == group_id_target]

        if len(matches) == 0:
            # Quintet NO trobat → ancorar a la força de l'equip, NO a 1.0.
            # (Aquest és el canvi clau que evita que les rotacions secundàries
            #  simulin com equips mitjans.)
            self.of_factor = float(np.clip(team_prior, *OF_CLIP))
            return

        # Quintet trobat: extreure dades empíriques
        lineup_row = matches.iloc[0]
        self.lineup_found = True
        self.lineup_minutes = float(lineup_row['MIN'])

        # Calcular ORtg del quintet:
        #   Possessions ≈ (MIN / 48) × PACE
        #   ORtg = PTS / Possessions × 100
        pace = float(lineup_row['PACE'])
        pts = float(lineup_row['PTS'])
        minutes = self.lineup_minutes

        if minutes > 0 and pace > 0:
            # Possessions del quintet en els minuts jugats
            # Nota: PACE és per 48 minuts d'un EQUIP (5 jugadors al mateix temps)
            possessions = (minutes / 48.0) * pace
            of_rating_lineup = (pts / possessions) * 100 if possessions > 0 else NBA_AVG_OF_RATING
        else:
            of_rating_lineup = NBA_AVG_OF_RATING

        # Acotar ORtg dins de rangs raonables NBA (90-135)
        of_rating_lineup = np.clip(of_rating_lineup, 90.0, 135.0)

        # Calcular alfa bayesià (shrinkage): α = N_0 / (N_0 + minutes)
        self.alpha_bayes = N_0 / (N_0 + minutes)

        # Factor empíric del quintet (raó respecte la mitjana NBA)
        empirical_factor = of_rating_lineup / NBA_AVG_OF_RATING

        # Combinació bayesiana cap al PRIOR DE L'EQUIP (abans era 1.0 neutre):
        #   of_factor = α · team_prior + (1 - α) · empirical_factor
        # Quan α = 1 (poques dades del quintet): of_factor ≈ força de l'equip
        # Quan α = 0 (moltes dades): of_factor ≈ eficiència real del quintet
        self.of_factor = self.alpha_bayes * team_prior + (1 - self.alpha_bayes) * empirical_factor
        self.of_factor = float(np.clip(self.of_factor, *OF_CLIP))

    def _estimate_team_off_factor(self, player_ids, nba_avg_ortg, clip_range):
        """
        MILLORA B: estima el factor ofensiu REAL de l'equip d'aquest quintet.

        Identifica a quin equip pertany el quintet per coincidència de jugadors
        al CSV (via TEAM_ID, exacte) i agrega TOTS els quintets d'aquest equip
        (ponderats per possessions) per obtenir l'ORtg de l'equip. Aquest valor
        substitueix el prior neutre 1.0 al shrinkage bayesià.

        Retorna 1.0 (neutre) si no es pot determinar l'equip.
        """
        df = self.lineups_df
        if df is None or len(df) == 0 or 'GROUP_ID' not in df.columns \
                or 'TEAM_ID' not in df.columns:
            return 1.0

        # Tokens delimitats per detectar cada jugador dins del GROUP_ID
        id_tokens = [f"-{pid}-" for pid in player_ids]

        def count_overlap(gid):
            if not isinstance(gid, str):
                return 0
            return sum(1 for t in id_tokens if t in gid)

        overlaps = df['GROUP_ID'].apply(count_overlap)
        if overlaps.max() == 0:
            return 1.0  # cap jugador trobat al CSV → no sabem l'equip

        # L'equip del quintet és el de la fila amb més jugadors nostres
        best_team_id = df.loc[overlaps.idxmax(), 'TEAM_ID']
        team_rows = df[df['TEAM_ID'] == best_team_id]
        if len(team_rows) == 0:
            return 1.0

        # ORtg de l'equip = punts totals / possessions totals (ponderat natural)
        try:
            pace = team_rows['PACE'].astype(float)
            mins = team_rows['MIN'].astype(float)
            pts = team_rows['PTS'].astype(float)
        except (KeyError, ValueError):
            return 1.0

        poss = (mins / 48.0) * pace
        total_poss = float(poss.sum())
        if total_poss <= 0:
            return 1.0

        team_ortg = (float(pts.sum()) / total_poss) * 100.0
        team_factor = team_ortg / nba_avg_ortg
        return float(np.clip(team_factor, *clip_range))
    
    def _calculate_opposing_defense(self):
        """
        Calcula l'impacte defensiu del rival utilitzant les seves defensive stats.
        
        Si el rival fa que els seus oponents tirin 44% (millor que la mitjana 47%),
        això afectarà negativament els percentatges d'aquest equip.
        """
        if self.opposing_lineup is None:
            # Sense rival, usar valors per defecte
            self.def_impact_2pt = 1.0
            self.def_impact_3pt = 1.0
            self.def_impact_overall = 1.0
        else:
            # Usar defensive stats del rival
            # Mitjana NBA: ~47% FG, ~49% 2P, ~36% 3P
            NBA_AVG_FG = 0.47
            NBA_AVG_2PT = 0.49
            NBA_AVG_3PT = 0.36
            
            rival_def_fg = self.opposing_lineup.avg_def_fg_pct_against
            rival_def_2pt = self.opposing_lineup.avg_def_2pt_pct_against
            rival_def_3pt = self.opposing_lineup.avg_def_3pt_pct_against
            
            # Calcular factor de modificació (BIDIRECCIONAL)
            # Si rival_def_fg = 0.44 (bona defensa), factor = 0.44/0.47 = 0.936 → Baixa percentatges ✅
            # Si rival_def_fg = 0.50 (mala defensa), factor = 0.50/0.47 = 1.064 → Puja percentatges ✅
            # Si rival_def_fg = 0.47 (mitjana NBA), factor = 1.0 → Cap canvi
            self.def_impact_overall = rival_def_fg / NBA_AVG_FG if rival_def_fg > 0 else 1.0
            self.def_impact_2pt = rival_def_2pt / NBA_AVG_2PT if rival_def_2pt > 0 else 1.0
            self.def_impact_3pt = rival_def_3pt / NBA_AVG_3PT if rival_def_3pt > 0 else 1.0
            
            # IMPACTE BIDIRECCIONAL EQUILIBRAT:
            # Rang ±7%: Permet que bona defensa baixi i mala defensa pugi percentatges
            # 
            # Exemples reals:
            # - Bona defensa (def_fg 44%): 0.44/0.47 = 0.936 → -6.4% ✅
            # - Defensa mitjana (def_fg 47%): 0.47/0.47 = 1.00 → 0% (neutral)
            # - Mala defensa (def_fg 50%): 0.50/0.47 = 1.064 → +6.4% ✅
            #
            # Efecte en shooter 40% des de 3PT:
            # - vs Bona defensa: 40% × 0.936 = 37.4% (-2.6%)
            # - vs Mala defensa: 40% × 1.064 = 42.6% (+2.6%)
            self.def_impact_overall = np.clip(self.def_impact_overall, 0.83, 1.17)
            self.def_impact_2pt = np.clip(self.def_impact_2pt, 0.83, 1.17)
            self.def_impact_3pt = np.clip(self.def_impact_3pt, 0.83, 1.17)
    
    def _build_transition_matrix(self) -> np.ndarray:
        """
        Construeix la matriu de transició 17x17 del QUINTET.

        Aquesta versió deriva matemàticament les taxes de falta i rebot ofensiu
        per zona a partir de les dades agregades del quintet, mantenint:
        - Defensa del rival (avg_defensive_*_block_pct)
        - Vulnerabilitat ofensiva (BLKA)
        - Impacte defensiu (def_impact_2pt, def_impact_3pt)

        FÓRMULA RIGOROSA per zona z:
            rate(z) = w(z) × k_quintet
        on:
            w(z)       = pes estructural de la zona
            k_quintet  = global_rate / Σ(freq(z) × w(z))
        """
        n = 17
        P = np.zeros((n, n))

        # ══════════════════════════════════════════════════════════════════
        # PESOS ESTRUCTURALS PER ZONA
        # ══════════════════════════════════════════════════════════════════
        ZONE_FOUL_WEIGHTS = {
            'corner_3':       0.02,
            'above_break_3':  0.025,
            'mid_range':      0.03,
            'long_mid_range': 0.03,
            'paint':          0.05,
            'layup':          0.17,
            'dunk':           0.25,
        }
        ZONE_OREB_WEIGHTS = {
            'corner_3':       0.40,
            'above_break_3':  0.40,
            'mid_range':      0.25,
            'long_mid_range': 0.25,
            'paint':          0.22,
            'layup':          0.40,
            'dunk':           0.40,
        }

        # ══════════════════════════════════════════════════════════════════
        # CÀLCUL DE TAXES PER ZONA - DERIVADES DEL QUINTET
        # ══════════════════════════════════════════════════════════════════
        frequencies = {
            'corner_3':       self.avg_corner_3_freq,
            'above_break_3':  self.avg_above_break_3_freq,
            'mid_range':      self.avg_mid_range_freq,
            'long_mid_range': self.avg_long_mid_range_freq,
            'paint':          self.avg_paint_freq,
            'layup':          self.avg_layup_freq,
            'dunk':           self.avg_dunk_freq,
        }

        # Taxes de falta per zona (derivades de avg_foul_draw_rate)
        foul_weighted_avg = sum(frequencies[z] * ZONE_FOUL_WEIGHTS[z]
                                for z in ZONE_FOUL_WEIGHTS)
        if foul_weighted_avg > 0 and self.avg_foul_draw_rate > 0:
            k_foul = self.avg_foul_draw_rate / foul_weighted_avg
            foul_rates = {z: min(0.95, max(0.005, ZONE_FOUL_WEIGHTS[z] * k_foul))
                          for z in ZONE_FOUL_WEIGHTS}
        else:
            foul_rates = {z: 0.02 for z in ZONE_FOUL_WEIGHTS}

        # Taxes de rebot ofensiu per zona (derivades de avg_offensive_reb_rate)
        oreb_global = self.avg_offensive_reb_rate
        if oreb_global < 0.05:
            # CSV mal escalat; reescalar per consistència amb NBA real
            oreb_global = min(0.5, oreb_global * 50)

        oreb_weighted_avg = sum(frequencies[z] * ZONE_OREB_WEIGHTS[z]
                                for z in ZONE_OREB_WEIGHTS)
        if oreb_weighted_avg > 0 and oreb_global > 0:
            k_oreb = oreb_global / oreb_weighted_avg
            oreb_rates = {z: min(0.50, max(0.05, ZONE_OREB_WEIGHTS[z] * k_oreb))
                          for z in ZONE_OREB_WEIGHTS}
        else:
            oreb_rates = {z: 0.25 for z in ZONE_OREB_WEIGHTS}

        # ══════════════════════════════════════════════════════════════════
        # START (0) - Selecció de zona de tir
        # ══════════════════════════════════════════════════════════════════
        total_threat = (self.avg_steal_rate + self.avg_deflection_rate) * 1.2
        safe_pct = max(0.75, 1.0 - self.avg_turnover_rate - total_threat)

        P[GameStates.START, GameStates.CORNER_3] = self.avg_corner_3_freq * safe_pct
        P[GameStates.START, GameStates.ABOVE_BREAK_3] = self.avg_above_break_3_freq * safe_pct
        P[GameStates.START, GameStates.MID_RANGE] = self.avg_mid_range_freq * safe_pct
        P[GameStates.START, GameStates.LONG_MID_RANGE] = self.avg_long_mid_range_freq * safe_pct
        P[GameStates.START, GameStates.PAINT] = self.avg_paint_freq * safe_pct
        P[GameStates.START, GameStates.LAYUP] = self.avg_layup_freq * safe_pct
        P[GameStates.START, GameStates.DUNK] = self.avg_dunk_freq * safe_pct

        P[GameStates.START, GameStates.TURNOVER] = self.avg_turnover_rate
        P[GameStates.START, GameStates.STEAL] = self.avg_steal_rate
        P[GameStates.START, GameStates.DEFLECTION] = self.avg_deflection_rate

        # ══════════════════════════════════════════════════════════════════
        # FUNCIÓ AUXILIAR PER CONFIGURAR UNA ZONA
        # (manté la lògica de defensa del rival)
        # ══════════════════════════════════════════════════════════════════
        def configure_zone(zone_state, zone_name, make_pct, block_prob_value):
            """Configura una zona del quintet amb taxes rigoroses."""
            # 1. Anotar
            make_rate = min(0.95, make_pct)
            miss_rate = 1.0 - make_rate

            # 2. Tap (passat externament - lògica defensa rival es manté fora)
            block_prob = min(0.30, block_prob_value)

            # 3. Falta (DERIVADA del avg_foul_draw_rate del quintet)
            foul_prob = foul_rates[zone_name]

            # 4. Garantir consistència
            events_no_make = block_prob + foul_prob
            if events_no_make > miss_rate * 0.95:
                scale = (miss_rate * 0.95) / events_no_make
                block_prob *= scale
                foul_prob *= scale

            # 5. Repartir la resta entre OREB i DREB
            remaining = max(0, miss_rate - block_prob - foul_prob)
            oreb_prob = oreb_rates[zone_name]

            # 6. Assignar
            P[zone_state, GameStates.END] = make_rate
            P[zone_state, GameStates.BLOCK] = block_prob
            P[zone_state, GameStates.FOUL] = foul_prob
            P[zone_state, GameStates.OFFENSIVE_REBOUND] = remaining * oreb_prob
            P[zone_state, GameStates.DEFENSIVE_REBOUND] = remaining * (1 - oreb_prob)

        # ══════════════════════════════════════════════════════════════════
        # CONFIGURAR ZONES AMB LA LÒGICA DE DEFENSA DEL RIVAL
        # NOU: També apliquem of_factor (calibració Bayesiana del quintet)
        # MILLORA A: i el home_court_factor (avantatge de pista local).
        # ══════════════════════════════════════════════════════════════════

        # Factor ofensiu efectiu = calibració del quintet × avantatge local
        of = self.of_factor * getattr(self, 'home_court_factor', 1.0)

        # CORNER 3
        if self.opposing_lineup:
            corner_3_block = self.opposing_lineup.avg_defensive_corner_3_block_pct / 100
        else:
            corner_3_block = 0.005
        configure_zone(GameStates.CORNER_3, 'corner_3',
                       self.avg_corner_3_pct * self.def_impact_3pt * of,
                       corner_3_block)

        # ABOVE BREAK 3
        if self.opposing_lineup:
            above_break_3_block = self.opposing_lineup.avg_defensive_above_break_3_block_pct / 100
        else:
            above_break_3_block = 0.005
        configure_zone(GameStates.ABOVE_BREAK_3, 'above_break_3',
                       self.avg_above_break_3_pct * self.def_impact_3pt * of,
                       above_break_3_block)

        # MID-RANGE
        if self.opposing_lineup:
            midrange_block = self.opposing_lineup.avg_defensive_midrange_block_pct / 100
        else:
            midrange_block = 0.03
        configure_zone(GameStates.MID_RANGE, 'mid_range',
                       self.avg_mid_range_pct * self.def_impact_2pt * of,
                       midrange_block)

        # LONG MID-RANGE
        if self.opposing_lineup:
            long_midrange_block = self.opposing_lineup.avg_defensive_long_midrange_block_pct / 100
        else:
            long_midrange_block = 0.02
        configure_zone(GameStates.LONG_MID_RANGE, 'long_mid_range',
                       self.avg_long_mid_range_pct * self.def_impact_2pt * of,
                       long_midrange_block)

        # PAINT (amb BLKA integrat)
        if self.opposing_lineup:
            opponent_defense = self.opposing_lineup.avg_defensive_paint_block_pct / 100
            paint_block = opponent_defense * self.avg_offensive_vulnerability
        else:
            paint_block = 0.06 * self.avg_offensive_vulnerability
        configure_zone(GameStates.PAINT, 'paint',
                       self.avg_paint_pct * self.def_impact_2pt * of,
                       paint_block)

        # LAYUP (amb BLKA integrat)
        if self.opposing_lineup:
            opponent_defense = self.opposing_lineup.avg_defensive_restricted_block_pct / 100
            restricted_block = opponent_defense * self.avg_offensive_vulnerability
        else:
            restricted_block = 0.08 * self.avg_offensive_vulnerability
        configure_zone(GameStates.LAYUP, 'layup',
                       self.avg_layup_pct * self.def_impact_2pt * of,
                       restricted_block)

        # DUNK
        if self.opposing_lineup:
            dunk_block = (self.opposing_lineup.avg_defensive_restricted_block_pct * 0.5) / 100
        else:
            dunk_block = 0.04
        configure_zone(GameStates.DUNK, 'dunk',
                       self.avg_dunk_pct * self.def_impact_2pt * of,
                       dunk_block)

        # ══════════════════════════════════════════════════════════════════
        # ESTATS FINALS I DEFENSIUS
        # ══════════════════════════════════════════════════════════════════
        P[GameStates.TURNOVER, GameStates.END] = 1.0
        P[GameStates.FOUL, GameStates.FREE_THROW] = 1.0
        P[GameStates.FREE_THROW, GameStates.END] = 1.0
        P[GameStates.STEAL, GameStates.END] = 1.0
        P[GameStates.BLOCK, GameStates.OFFENSIVE_REBOUND] = 0.3
        P[GameStates.BLOCK, GameStates.DEFENSIVE_REBOUND] = 0.7
        P[GameStates.DEFLECTION, GameStates.TURNOVER] = 0.6
        P[GameStates.DEFLECTION, GameStates.START] = 0.4
        P[GameStates.OFFENSIVE_REBOUND, GameStates.START] = 1.0
        P[GameStates.DEFENSIVE_REBOUND, GameStates.END] = 1.0
        P[GameStates.END, GameStates.END] = 1.0

        # Normalitzar cada fila
        for i in range(n):
            row_sum = P[i].sum()
            if row_sum > 0:
                P[i] /= row_sum
            else:
                P[i, GameStates.END] = 1.0

        return P
    
    def simulate_possession(self, verbose: bool = False) -> Tuple[List[int], int, Dict]:
        """Simula una possessió."""
        return self.markov_chain.simulate_possession(verbose=verbose)


class AdvancedMarkovChain:
    """Cadena de Markov avançada."""
    
    def __init__(self, transition_matrix: np.ndarray):
        self.P = transition_matrix
        self.n_states = len(GameStates.STATE_NAMES)
        self.validate_matrix()
    
    def validate_matrix(self):
        """Valida la matriu."""
        assert self.P.shape == (self.n_states, self.n_states)
        assert (self.P >= 0).all()
        row_sums = self.P.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), f"Files no sumen 1: {row_sums}"
    
    def simulate_possession(self, initial_state: int = GameStates.START,
                          max_steps: int = 50,
                          verbose: bool = False) -> Tuple[List[int], int, Dict]:
        """
        Simula una possessió completa.
        
        Returns:
            path: Seqüència d'estats
            points: Punts anotats
            stats: Estadístiques de la possessió
        """
        path = [initial_state]
        current_state = initial_state
        points = 0
        steps = 0
        
        stats = {
            'shot_attempts': 0,
            'shots_made': 0,
            'two_point_attempts': 0,
            'two_point_made': 0,
            'three_point_attempts': 0,
            'three_point_made': 0,
            'free_throw_attempts': 0,
            'free_throw_made': 0,
            'turnovers': 0,
            'steals': 0,
            'blocks': 0,
            'deflections': 0,
            'offensive_rebounds': 0,
            'defensive_rebounds': 0,
            'fouls_drawn': 0
        }
        
        if verbose:
            print(f"Inici: {GameStates.STATE_NAMES[current_state]}")
        
        while current_state != GameStates.END and steps < max_steps:
            next_state = np.random.choice(self.n_states, p=self.P[current_state])
            
            # Calcular punts i stats
            points_scored, stats = self._process_transition(
                current_state, next_state, stats
            )
            points += points_scored
            
            if verbose:
                print(f"  → {GameStates.STATE_NAMES[next_state]}" + 
                     (f" (+{points_scored} pts)" if points_scored > 0 else ""))
            
            path.append(next_state)
            current_state = next_state
            steps += 1
        
        return path, points, stats
    
    def _process_transition(self, from_state: int, to_state: int,
                          stats: Dict) -> Tuple[int, Dict]:
        """Processa una transició i actualitza estadístiques."""
        points = 0
        
        # CORNER 3
        if from_state == GameStates.CORNER_3:
            stats['shot_attempts'] += 1
            stats['three_point_attempts'] += 1
            
            if to_state == GameStates.END:
                points = 3
                stats['shots_made'] += 1
                stats['three_point_made'] += 1
            elif to_state == GameStates.FOUL:
                if np.random.random() < 0.25:  # And-one 25%
                    points = 3
                    stats['shots_made'] += 1
                    stats['three_point_made'] += 1
        
        # ABOVE BREAK 3
        elif from_state == GameStates.ABOVE_BREAK_3:
            stats['shot_attempts'] += 1
            stats['three_point_attempts'] += 1
            
            if to_state == GameStates.END:
                points = 3
                stats['shots_made'] += 1
                stats['three_point_made'] += 1
            elif to_state == GameStates.FOUL:
                if np.random.random() < 0.25:  # And-one 25%
                    points = 3
                    stats['shots_made'] += 1
                    stats['three_point_made'] += 1
        
        # MID-RANGE
        elif from_state == GameStates.MID_RANGE:
            stats['shot_attempts'] += 1
            stats['two_point_attempts'] += 1
            
            if to_state == GameStates.END:
                points = 2
                stats['shots_made'] += 1
                stats['two_point_made'] += 1
            elif to_state == GameStates.FOUL:
                if np.random.random() < 0.45:  # And-one 45%
                    points = 2
                    stats['shots_made'] += 1
                    stats['two_point_made'] += 1
        
        # LONG MID-RANGE
        elif from_state == GameStates.LONG_MID_RANGE:
            stats['shot_attempts'] += 1
            stats['two_point_attempts'] += 1
            
            if to_state == GameStates.END:
                points = 2
                stats['shots_made'] += 1
                stats['two_point_made'] += 1
            elif to_state == GameStates.FOUL:
                if np.random.random() < 0.40:  # And-one 40%
                    points = 2
                    stats['shots_made'] += 1
                    stats['two_point_made'] += 1
        
        # PAINT
        elif from_state == GameStates.PAINT:
            stats['shot_attempts'] += 1
            stats['two_point_attempts'] += 1
            
            if to_state == GameStates.END:
                points = 2
                stats['shots_made'] += 1
                stats['two_point_made'] += 1
            elif to_state == GameStates.FOUL:
                if np.random.random() < 0.55:  # And-one 55%
                    points = 2
                    stats['shots_made'] += 1
                    stats['two_point_made'] += 1
        
        # LAYUP
        elif from_state == GameStates.LAYUP:
            stats['shot_attempts'] += 1
            stats['two_point_attempts'] += 1
            
            if to_state == GameStates.END:
                points = 2
                stats['shots_made'] += 1
                stats['two_point_made'] += 1
            elif to_state == GameStates.FOUL:
                if np.random.random() < 0.60:  # And-one 60%
                    points = 2
                    stats['shots_made'] += 1
                    stats['two_point_made'] += 1
        
        # DUNK
        elif from_state == GameStates.DUNK:
            stats['shot_attempts'] += 1
            stats['two_point_attempts'] += 1
            
            if to_state == GameStates.END:
                points = 2
                stats['shots_made'] += 1
                stats['two_point_made'] += 1
            elif to_state == GameStates.FOUL:
                if np.random.random() < 0.70:  # And-one 70%
                    points = 2
                    stats['shots_made'] += 1
                    stats['two_point_made'] += 1
        
        # FREE THROW
        elif from_state == GameStates.FREE_THROW:
            # 2 tirs lliures amb 78% d'encert
            for _ in range(2):
                stats['free_throw_attempts'] += 1
                if np.random.random() < 0.78:
                    points += 1
                    stats['free_throw_made'] += 1
        
        # Estadístiques defensives
        if to_state == GameStates.TURNOVER:
            stats['turnovers'] += 1
        elif to_state == GameStates.STEAL:
            stats['steals'] += 1
            stats['turnovers'] += 1
        elif to_state == GameStates.BLOCK:
            stats['blocks'] += 1
        elif to_state == GameStates.DEFLECTION:
            stats['deflections'] += 1
        elif to_state == GameStates.OFFENSIVE_REBOUND:
            stats['offensive_rebounds'] += 1
        elif to_state == GameStates.DEFENSIVE_REBOUND:
            stats['defensive_rebounds'] += 1
        elif to_state == GameStates.FOUL:
            stats['fouls_drawn'] += 1
        
        return points, stats


# ===================
# FUNCIONS D'UTILITAT
# ===================

def load_scraped_data(csv_path: str = 'nba_data_advanced/advanced_player_data.csv') -> pd.DataFrame:
    """
    Carrega les dades del scraper.
    
    Args:
        csv_path: Camí al CSV amb les dades
        
    Returns:
        DataFrame amb totes les dades
    """
    return pd.read_csv(csv_path)


def create_player_from_data(df: pd.DataFrame, player_name: str) -> AdvancedPlayerFromData:
    """
    Crea un jugador a partir de les dades del DataFrame (CSV del scraper).
    
    Args:
        df: DataFrame carregat des de 'advanced_player_data.csv'
        player_name: Nom del jugador (o part del nom)
        
    Returns:
        Objecte AdvancedPlayerFromData amb dades reals
    """
    # Buscar jugador
    matches = df[df['player_name'].str.contains(player_name, case=False, na=False)]
    
    if len(matches) == 0:
        raise ValueError(f"Jugador '{player_name}' no trobat al CSV")
    
    if len(matches) > 1:
        print(f"⚠️  Trobats {len(matches)} jugadors amb '{player_name}':")
        for _, p in matches.iterrows():
            print(f"  • {p['player_name']}")
        raise ValueError("Especifica millor el nom del jugador")
    
    # Obtenir fila del jugador
    player_row = matches.iloc[0]
    
    # Construir diccionari amb totes les dades
    player_data = {
        'player_id': player_row.get('player_id'),
        'player_name': player_row.get('player_name'),
        
        # Shot chart raw (si existeix com a JSON string)
        'shot_chart_raw': player_row.get('shot_chart_raw'),
        
        # Percentatges per zona (calculats pel scraper)
        'jumpshot_2p_pct': player_row.get('jumpshot_2p_pct', 0),
        'jumpshot_3p_pct': player_row.get('jumpshot_3p_pct', 0),
        'layup_pct': player_row.get('layup_pct', 0),
        'dunk_pct': player_row.get('dunk_pct', 0),
        
        # Freqüències per zona
        'jumpshot_2p_freq': player_row.get('jumpshot_2p_freq', 0),
        'jumpshot_3p_freq': player_row.get('jumpshot_3p_freq', 0),
        'layup_freq': player_row.get('layup_freq', 0),
        'dunk_freq': player_row.get('dunk_freq', 0),
        
        # Estadístiques defensives
        'def_fga_against': player_row.get('def_fga_against', 0),
        'def_fgm_against': player_row.get('def_fgm_against', 0),
        'def_fg_pct_against': player_row.get('def_fg_pct_against', 0.47),
        'def_2pt_pct_against': player_row.get('def_2pt_pct_against', 0.49),
        'def_3pt_pct_against': player_row.get('def_3pt_pct_against', 0.36),
        
        # Hustle stats
        'deflections': player_row.get('deflections', 0),
        'charges_drawn': player_row.get('charges_drawn', 0),
        'screen_assists': player_row.get('screen_assists', 0),
        'loose_balls_recovered': player_row.get('loose_balls_recovered', 0),
        'contested_shots': player_row.get('contested_shots', 0),
    }
    
    return AdvancedPlayerFromData(player_data)


def create_lineup_from_names(df: pd.DataFrame, player_names: List[str], 
                            lineup_name: str = "Custom Lineup") -> AdvancedLineupFromData:
    """
    Crea un quintet a partir de noms de jugadors.
    
    Args:
        df: DataFrame amb dades
        player_names: Llista amb 5 noms de jugadors
        lineup_name: Nom del quintet
        
    Returns:
        Objecte AdvancedLineupFromData
    """
    players = []
    
    for name in player_names:
        try:
            player = create_player_from_data(df, name)
            players.append(player)
            print(f"✓ {player.name}")
        except Exception as e:
            print(f"✗ Error amb '{name}': {e}")
    
    if len(players) != 5:
        raise ValueError(f"Cal exactament 5 jugadors, només s'han trobat {len(players)}")
    
    return AdvancedLineupFromData(lineup_name, players)


def print_lineup_info(lineup: AdvancedLineupFromData):
    """Mostra informació detallada del quintet."""
    print(f"\n{'='*70}")
    print(f"QUINTET: {lineup.name}")
    print(f"{'='*70}")
    
    print(f"\n👥 Jugadors:")
    for i, player in enumerate(lineup.players, 1):
        print(f"\n  {i}. {player.name}")
        print(f"     Shot Chart:")
        print(f"     - Corner 3: {player.corner_3_pct:.1%} (freq: {player.corner_3_freq:.1%})")
        print(f"     - Above Break 3: {player.above_break_3_pct:.1%} (freq: {player.above_break_3_freq:.1%})")
        print(f"     - Mid-Range: {player.mid_range_pct:.1%} (freq: {player.mid_range_freq:.1%})")
        print(f"     - Long Mid-Range: {player.long_mid_range_pct:.1%} (freq: {player.long_mid_range_freq:.1%})")
        print(f"     - Paint: {player.paint_pct:.1%} (freq: {player.paint_freq:.1%})")
        print(f"     - Layup: {player.layup_pct:.1%} (freq: {player.layup_freq:.1%})")
        print(f"     - Dunk: {player.dunk_pct:.1%} (freq: {player.dunk_freq:.1%})")
        print(f"     Defensa:")
        print(f"     - FG% rivals: {player.def_fg_pct_against:.1%}")
        print(f"     - Deflections: {player.deflections:.0f}")
    
    print(f"\n📊 Mitjanes del Quintet:")
    print(f"   Corner 3: {lineup.avg_corner_3_pct:.1%}")
    print(f"   Above Break 3: {lineup.avg_above_break_3_pct:.1%}")
    print(f"   Mid-Range: {lineup.avg_mid_range_pct:.1%}")
    print(f"   Long Mid-Range: {lineup.avg_long_mid_range_pct:.1%}")
    print(f"   Paint: {lineup.avg_paint_pct:.1%}")
    print(f"   Layup: {lineup.avg_layup_pct:.1%}")
    print(f"   Dunk: {lineup.avg_dunk_pct:.1%}")
    
    print(f"\n🛡️  Impacte Defensiu:")
    print(f"   Defensive Impact 2PT: {lineup.def_impact_2pt:.3f}")
    print(f"   Defensive Impact 3PT: {lineup.def_impact_3pt:.3f}")
    print(f"   Overall: {lineup.def_impact_overall:.3f}")


# ===================
# EXEMPLE D'ÚS
# ===================

if __name__ == "__main__":
    print("\n🏀 MODEL AVANÇAT NBA AMB DADES REALS\n")
    
    # 1. Carregar dades
    print("📊 Carregant dades del scraper...")
    df = load_scraped_data()
    print(f"   ✓ {len(df)} jugadors carregats\n")
    
    # 2. Crear jugadors
    print("👥 Creant jugadors...")
    try:
        lebron = create_player_from_data(df, "LeBron")
        print(f"   ✓ {lebron.name}")
        print(f"      Corner 3: {lebron.corner_3_pct:.1%} | Above Break 3: {lebron.above_break_3_pct:.1%}")
        print(f"      Mid-Range: {lebron.mid_range_pct:.1%} | Layup: {lebron.layup_pct:.1%} | Dunk: {lebron.dunk_pct:.1%}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("   ℹ️  Assegura't que has executat el scraper primer!")
        sys.exit(1)
    
    # 3. Crear quintet
    print("\n🏀 Creant quintet Lakers...")
    lakers_names = ["LeBron", "Anthony Davis", "Austin Reaves", "D'Angelo Russell", "Rui"]
    
    try:
        lakers = create_lineup_from_names(df, lakers_names, "Lakers Starter")
        print_lineup_info(lakers)
    except Exception as e:
        print(f"✗ Error creant quintet: {e}")
        sys.exit(1)
    
    # 4. Simular una possessió
    print(f"\n🎮 Simulant una possessió amb el quintet Lakers...\n")
    path, points, stats = lakers.simulate_possession(verbose=True)
    
    print(f"\n📈 Resultats:")
    print(f"   Punts: {points}")
    print(f"   Tirs: {stats['shots_made']}/{stats['shot_attempts']}")
    print(f"   Pèrdues: {stats['turnovers']}")
    print(f"   Rebots ofensius: {stats['offensive_rebounds']}")
    
    # 5. Simular múltiples possessions
    print(f"\n🔄 Simulant 100 possessions...")
    total_points = 0
    total_stats = {
        'shot_attempts': 0, 'shots_made': 0,
        'turnovers': 0, 'offensive_rebounds': 0
    }
    
    for _ in range(100):
        _, pts, st = lakers.simulate_possession()
        total_points += pts
        for key in total_stats:
            total_stats[key] += st[key]
    
    print(f"\n📊 Estadístiques de 100 possessions:")
    print(f"   Punts totals: {total_points}")
    print(f"   Punts/possessió: {total_points/100:.2f}")
    print(f"   Offensive Rating: {total_points:.1f}")
    print(f"   FG%: {total_stats['shots_made']/total_stats['shot_attempts']*100:.1f}%")
    print(f"   Turnover rate: {total_stats['turnovers']/100:.2f}")
    print(f"   Rebot ofensiu rate: {total_stats['offensive_rebounds']/100:.2f}")
    
    print("\n✅ Model carregat i funcionant!")
    print("\nPer usar-lo:")
    print("  1. Carrega dades: df = load_scraped_data()")
    print("  2. Crea jugadors: player = create_player_from_data(df, 'Nom')")
    print("  3. Crea quintet: lineup = create_lineup_from_names(df, [noms])")
    print("  4. Simula: path, points, stats = lineup.simulate_possession()")


# ========================================
# SIMULACIÓ DE PARTIT COMPLET AMB OVERTIME
# ========================================

def simulate_game(home_lineup: AdvancedLineupFromData, 
                 away_lineup: AdvancedLineupFromData,
                 quarters: int = 4,
                 possessions_per_quarter: int = 50,
                 verbose: bool = True) -> Dict:
    """
    Simula un partit complet entre dos quintets.
    Si hi ha empat, es juga OVERTIME fins que hi hagi guanyador.
    
    Args:
        home_lineup: Quintet local
        away_lineup: Quintet visitant
        quarters: Número de quarters (4 per defecte)
        possessions_per_quarter: Possessions per quarter (~50 = 12 minuts)
        verbose: Mostrar progrés
    
    Returns:
        Diccionari amb:
        - 'home_score': Punts equip local
        - 'away_score': Punts equip visitant
        - 'winner': 'home' o 'away'
        - 'overtime_periods': Número de pròrrogues (0 si no n'hi ha)
        - 'home_stats': Estadístiques locals
        - 'away_stats': Estadístiques visitants
        - 'quarter_scores': Punts per quarter
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🏀 {home_lineup.name.upper()} vs {away_lineup.name.upper()}")
        print(f"{'='*70}\n")
    
    # Estadístiques totals
    home_score = 0
    away_score = 0
    home_stats = {
        'shot_attempts': 0, 'shots_made': 0,
        'free_throw_attempts': 0, 'free_throw_made': 0,
        'turnovers': 0, 'steals': 0, 'blocks': 0,
        'offensive_rebounds': 0, 'defensive_rebounds': 0,
        'deflections': 0, 'fouls_drawn': 0
    }
    away_stats = home_stats.copy()
    
    quarter_scores = []
    
    # ==================
    # TEMPS REGLAMENTARI (4 quarters)
    # ==================
    for q in range(1, quarters + 1):
        if verbose:
            print(f"🕐 QUARTER {q}")
            print("-" * 70)
        
        q_home = 0
        q_away = 0
        
        # Simular possessions alternades
        for i in range(possessions_per_quarter):
            # Possessió local
            _, pts, stats = home_lineup.simulate_possession()
            home_score += pts
            q_home += pts
            for key in home_stats:
                home_stats[key] += stats[key]
            
            # Possessió visitant
            _, pts, stats = away_lineup.simulate_possession()
            away_score += pts
            q_away += pts
            for key in away_stats:
                away_stats[key] += stats[key]
        
        quarter_scores.append({
            'quarter': q,
            'home': q_home,
            'away': q_away
        })
        
        if verbose:
            print(f"   {home_lineup.name}: {q_home} punts")
            print(f"   {away_lineup.name}: {q_away} punts")
            print(f"   Marcador: {home_lineup.name} {home_score} - {away_score} {away_lineup.name}\n")
    
    # ==================
    # OVERTIME (si hi ha empat)
    # ==================
    overtime_periods = 0
    
    while home_score == away_score:
        overtime_periods += 1
        
        if verbose:
            print(f"⏱️  OVERTIME {overtime_periods}")
            print("-" * 70)
        
        ot_home = 0
        ot_away = 0
        
        # Overtime = ~25 possessions (5 minuts)
        ot_possessions = 25
        
        for i in range(ot_possessions):
            # Possessió local
            _, pts, stats = home_lineup.simulate_possession()
            home_score += pts
            ot_home += pts
            for key in home_stats:
                home_stats[key] += stats[key]
            
            # Possessió visitant
            _, pts, stats = away_lineup.simulate_possession()
            away_score += pts
            ot_away += pts
            for key in away_stats:
                away_stats[key] += stats[key]
        
        quarter_scores.append({
            'quarter': f'OT{overtime_periods}',
            'home': ot_home,
            'away': ot_away
        })
        
        if verbose:
            print(f"   {home_lineup.name}: {ot_home} punts")
            print(f"   {away_lineup.name}: {ot_away} punts")
            print(f"   Marcador: {home_lineup.name} {home_score} - {away_score} {away_lineup.name}\n")
        
        # IMPORTANT: Si encara empaten després de OT, continuar
        # En la NBA real, continuen jugant OTs fins que algú guanyi
    
    # ==================
    # RESULTATS FINALS
    # ==================
    winner = 'home' if home_score > away_score else 'away'
    winner_name = home_lineup.name if winner == 'home' else away_lineup.name
    loser_name = away_lineup.name if winner == 'home' else home_lineup.name
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🏆 RESULTAT FINAL")
        print(f"{'='*70}")
        print(f"\n   {winner_name.upper()} {max(home_score, away_score)} - {min(home_score, away_score)} {loser_name.upper()}")
        
        if overtime_periods > 0:
            print(f"\n   ⏱️  Pròrrogues: {overtime_periods}")
        
        print(f"\n📊 ESTADÍSTIQUES:")
        print(f"\n   {home_lineup.name}:")
        print(f"      FG: {home_stats['shots_made']}/{home_stats['shot_attempts']} ({home_stats['shots_made']/max(1,home_stats['shot_attempts'])*100:.1f}%)")
        print(f"      FT: {home_stats['free_throw_made']}/{home_stats['free_throw_attempts']}")
        print(f"      Rebots: {home_stats['offensive_rebounds']} OFF, {home_stats['defensive_rebounds']} DEF")
        print(f"      TOV: {home_stats['turnovers']} | STL: {home_stats['steals']} | BLK: {home_stats['blocks']}")
        
        print(f"\n   {away_lineup.name}:")
        print(f"      FG: {away_stats['shots_made']}/{away_stats['shot_attempts']} ({away_stats['shots_made']/max(1,away_stats['shot_attempts'])*100:.1f}%)")
        print(f"      FT: {away_stats['free_throw_made']}/{away_stats['free_throw_attempts']}")
        print(f"      Rebots: {away_stats['offensive_rebounds']} OFF, {away_stats['defensive_rebounds']} DEF")
        print(f"      TOV: {away_stats['turnovers']} | STL: {away_stats['steals']} | BLK: {away_stats['blocks']}")
        print(f"\n{'='*70}\n")
    
    return {
        'home_score': home_score,
        'away_score': away_score,
        'winner': winner,
        'winner_name': winner_name,
        'overtime_periods': overtime_periods,
        'home_stats': home_stats,
        'away_stats': away_stats,
        'quarter_scores': quarter_scores
    }


def simulate_multiple_games(home_lineup: AdvancedLineupFromData,
                           away_lineup: AdvancedLineupFromData,
                           num_games: int = 10,
                           verbose: bool = False) -> Dict:
    """
    Simula múltiples partits i retorna estadístiques agregades.
    
    Args:
        home_lineup: Quintet local
        away_lineup: Quintet visitant
        num_games: Número de partits a simular
        verbose: Mostrar cada partit
    
    Returns:
        Diccionari amb estadístiques agregades
    """
    results = {
        'home_wins': 0,
        'away_wins': 0,
        'total_overtimes': 0,
        'avg_home_score': 0,
        'avg_away_score': 0,
        'games': []
    }
    
    print(f"\n🎮 Simulant {num_games} partits: {home_lineup.name} vs {away_lineup.name}")
    print("=" * 70)
    
    for i in range(num_games):
        game = simulate_game(home_lineup, away_lineup, verbose=verbose)
        
        results['games'].append(game)
        results['avg_home_score'] += game['home_score']
        results['avg_away_score'] += game['away_score']
        results['total_overtimes'] += game['overtime_periods']
        
        if game['winner'] == 'home':
            results['home_wins'] += 1
        else:
            results['away_wins'] += 1
        
        if not verbose:
            ot_text = f" (OT{game['overtime_periods']})" if game['overtime_periods'] > 0 else ""
            print(f"   Partit {i+1}: {home_lineup.name} {game['home_score']} - {game['away_score']} {away_lineup.name}{ot_text}")
    
    results['avg_home_score'] /= num_games
    results['avg_away_score'] /= num_games
    results['home_win_pct'] = results['home_wins'] / num_games
    
    print(f"\n{'='*70}")
    print(f"📊 RESUM DE {num_games} PARTITS")
    print(f"{'='*70}")
    print(f"\n   Victòries:")
    print(f"      {home_lineup.name}: {results['home_wins']} ({results['home_win_pct']*100:.1f}%)")
    print(f"      {away_lineup.name}: {results['away_wins']} ({(1-results['home_win_pct'])*100:.1f}%)")
    print(f"\n   Puntuació mitjana:")
    print(f"      {home_lineup.name}: {results['avg_home_score']:.1f}")
    print(f"      {away_lineup.name}: {results['avg_away_score']:.1f}")
    print(f"\n   Pròrrogues totals: {results['total_overtimes']}")
    print(f"   Pròrrogues per partit: {results['total_overtimes']/num_games:.2f}")
    print(f"\n{'='*70}\n")
    
    return results