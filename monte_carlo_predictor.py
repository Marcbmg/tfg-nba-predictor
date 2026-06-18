"""
Sistema de Predicció de Partits NBA
====================================

Utilitza simulació Monte Carlo amb cadenes de Markov per:
1. Predir resultats de partits
2. Calcular probabilitats de victòria
3. Predir marcadors esperats
4. Analitzar distribucions de puntuació
5. Generar intervals de confiança
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
from realistic_game_simulator import RealisticGameSimulator, load_team_rotation
from complete_markov_model import *


class MonteCarloGamePredictor:
    """
    Predictor de partits usant simulació Monte Carlo.
    """
    
    def __init__(self, team_a_rotation, team_b_rotation, home_team: str = 'a',
                 home_court_pts: float = 2.8):
        """
        Args:
            team_a_rotation: Rotació equip A
            team_b_rotation: Rotació equip B
            home_team: Equip local ('a', 'b' o None). Per defecte 'a'.
            home_court_pts: Magnitud de l'avantatge local en punts (NBA ≈ 2.8).
        """
        self.team_a = team_a_rotation
        self.team_b = team_b_rotation
        # MILLORA A: l'avantatge local ara es modela DINS la simulació
        # (via eficiència + pace), no com a punts plans al final.
        self.simulator = RealisticGameSimulator(
            team_a_rotation, team_b_rotation,
            home_team=home_team, home_court_pts=home_court_pts
        )
        self.simulation_results = []
    
    def run_monte_carlo(self, n_simulations: int = 1000, 
                       verbose: bool = True,
                       home_advantage: float = 0.0) -> Dict:
        """
        Executa simulació Monte Carlo de n partits.
        
        Args:
            n_simulations: Número de simulacions (recomanat: 200-1000)
            verbose: Mostrar progrés
            home_advantage: (OBSOLET) Punts plans extra per l'equip A. A partir
                de la MILLORA A l'avantatge local es modela dins la simulació
                (eficiència + pace), així que es recomana deixar-ho a 0.0 per
                no comptar-lo dues vegades. Es manté per compatibilitat.
            
        Returns:
            Diccionari amb estadístiques de predicció
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🎲 SIMULACIÓ MONTE CARLO")
            print(f"{'='*70}\n")
            print(f"Simulant {n_simulations:,} partits...")
            print(f"{self.team_a.team_name} vs {self.team_b.team_name}")
            if home_advantage > 0:
                print(f"Home Court Advantage: +{home_advantage:.1f} pts per {self.team_a.team_name}\n")
            else:
                print()
        
        self.simulation_results = []
        
        # Simular n partits
        for i in range(n_simulations):
            if verbose and (i + 1) % 100 == 0:
                print(f"   Progrés: {i+1:,}/{n_simulations:,} ({(i+1)/n_simulations*100:.1f}%)")
            
            # Simular partit (silenciós)
            game = self.simulator.simulate_game(
                verbose=False,
                show_quarters=False,
                show_key_moments=False
            )
            
            # Aplicar home court advantage
            adjusted_score_a = game['score_a'] + home_advantage
            adjusted_score_b = game['score_b']
            
            self.simulation_results.append({
                'score_a': adjusted_score_a,
                'score_b': adjusted_score_b,
                'winner': self.team_a.team_name if adjusted_score_a > adjusted_score_b else self.team_b.team_name,
                'margin': abs(adjusted_score_a - adjusted_score_b),
                'total_points': adjusted_score_a + adjusted_score_b,
                'possessions_a': game['possessions_a'],
                'possessions_b': game['possessions_b'],
                'pace': game['pace'],
                'ortg_a': game['ortg_a'],
                'ortg_b': game['ortg_b']
            })
        
        # Calcular estadístiques
        stats = self._calculate_statistics()
        stats['home_advantage'] = home_advantage  # Guardar per referència
        
        if verbose:
            print(f"\n✅ {n_simulations:,} simulacions completades!\n")
        
        return stats
    
    def _calculate_statistics(self) -> Dict:
        """Calcula estadístiques de les simulacions."""
        
        results = self.simulation_results
        
        # Convertir a arrays
        scores_a = np.array([r['score_a'] for r in results])
        scores_b = np.array([r['score_b'] for r in results])
        margins = np.array([r['margin'] for r in results])
        totals = np.array([r['total_points'] for r in results])
        winners = [r['winner'] for r in results]
        
        # NUEVO: Tracking de possessions
        possessions_a = np.array([r['possessions_a'] for r in results])
        possessions_b = np.array([r['possessions_b'] for r in results])
        paces = np.array([r['pace'] for r in results])
        
        # Probabilitats de victòria
        # IMPORTANT: winner ara és el nom de l'equip, no 'A' o 'B'
        wins_a = sum(1 for w in winners if w == self.team_a.team_name)
        wins_b = sum(1 for w in winners if w == self.team_b.team_name)
        ties = sum(1 for w in winners if w == 'Tie')
        
        prob_a = wins_a / len(results)
        prob_b = wins_b / len(results)
        prob_tie = ties / len(results)
        
        # Estadístiques descriptives
        stats = {
            'n_simulations': len(results),
            'team_a': self.team_a.team_name,
            'team_b': self.team_b.team_name,
            
            # Probabilitats
            'prob_win_a': prob_a,
            'prob_win_b': prob_b,
            'prob_tie': prob_tie,
            'wins_a': wins_a,
            'wins_b': wins_b,
            'ties': ties,
            
            # Marcadors esperats
            'expected_score_a': np.mean(scores_a),
            'expected_score_b': np.mean(scores_b),
            'std_score_a': np.std(scores_a),
            'std_score_b': np.std(scores_b),
            
            # Intervals de confiança (95%)
            'ci_95_score_a': (np.percentile(scores_a, 2.5), np.percentile(scores_a, 97.5)),
            'ci_95_score_b': (np.percentile(scores_b, 2.5), np.percentile(scores_b, 97.5)),
            
            # Marge esperat
            'expected_margin_a': np.mean(scores_a - scores_b),
            'expected_margin_abs': np.mean(margins),
            'std_margin': np.std(scores_a - scores_b),
            
            # Total de punts
            'expected_total': np.mean(totals),
            'std_total': np.std(totals),
            'ci_95_total': (np.percentile(totals, 2.5), np.percentile(totals, 97.5)),
            
            # NUEVO: Possessions i Pace
            'expected_possessions_a': np.mean(possessions_a),
            'expected_possessions_b': np.mean(possessions_b),
            'std_possessions_a': np.std(possessions_a),
            'std_possessions_b': np.std(possessions_b),
            'expected_pace': np.mean(paces),
            'std_pace': np.std(paces),
            'ci_95_pace': (np.percentile(paces, 2.5), np.percentile(paces, 97.5)),
            
            # Over/Under
            'prob_over_220': sum(1 for t in totals if t > 220) / len(totals),
            'prob_under_220': sum(1 for t in totals if t <= 220) / len(totals),
            
            # Distribucions
            'score_distribution_a': scores_a,
            'score_distribution_b': scores_b,
            'margin_distribution': scores_a - scores_b,
            'total_distribution': totals,
            'possessions_distribution_a': possessions_a,
            'possessions_distribution_b': possessions_b,
            'pace_distribution': paces
        }
        
        return stats
    
    def print_prediction(self, stats: Dict):
        """Mostra predicció de forma llegible."""
        
        print(f"\n{'='*70}")
        print(f"📊 PREDICCIÓ DEL PARTIT")
        print(f"{'='*70}\n")
        
        print(f"🏀 {stats['team_a']} vs {stats['team_b']}")
        print(f"   Basat en {stats['n_simulations']:,} simulacions Monte Carlo\n")
        
        # Probabilitats de victòria
        print(f"🎯 PROBABILITATS DE VICTÒRIA:\n")
        
        prob_a = stats['prob_win_a']
        prob_b = stats['prob_win_b']
        
        # Barra visual
        bar_length = 50
        bar_a = int(prob_a * bar_length)
        bar_b = int(prob_b * bar_length)
        
        print(f"   {stats['team_a'][:25]:25s} {prob_a*100:5.1f}% {'█' * bar_a}")
        print(f"   {stats['team_b'][:25]:25s} {prob_b*100:5.1f}% {'█' * bar_b}")
        
        if stats['prob_tie'] > 0.001:
            print(f"   {'Empat':25s} {stats['prob_tie']*100:5.1f}%")
        
        print(f"\n   Victories simulades:")
        print(f"      {stats['team_a']}: {stats['wins_a']:,}/{stats['n_simulations']:,}")
        print(f"      {stats['team_b']}: {stats['wins_b']:,}/{stats['n_simulations']:,}")
        
        # Favorit
        if prob_a > prob_b:
            favorit = stats['team_a']
            prob_fav = prob_a
        else:
            favorit = stats['team_b']
            prob_fav = prob_b
        
        print(f"\n   🏆 Favorit: {favorit} ({prob_fav*100:.1f}%)")
        
        # Marcadors esperats
        print(f"\n📈 MARCADORS ESPERATS:\n")
        
        exp_a = stats['expected_score_a']
        exp_b = stats['expected_score_b']
        ci_a = stats['ci_95_score_a']
        ci_b = stats['ci_95_score_b']
        
        print(f"   {stats['team_a']:25s} {exp_a:5.1f} ± {stats['std_score_a']:4.1f}")
        print(f"      Interval 95%: [{ci_a[0]:.1f}, {ci_a[1]:.1f}]")
        
        print(f"\n   {stats['team_b']:25s} {exp_b:5.1f} ± {stats['std_score_b']:4.1f}")
        print(f"      Interval 95%: [{ci_b[0]:.1f}, {ci_b[1]:.1f}]")
        
        print(f"\n   📊 Predicció final: {exp_a:.0f} - {exp_b:.0f}")
        
        # Marge esperat
        print(f"\n🎲 MARGE ESPERAT:\n")
        
        margin = stats['expected_margin_a']
        if margin > 0:
            print(f"   {stats['team_a']} per {abs(margin):.1f} punts")
        elif margin < 0:
            print(f"   {stats['team_b']} per {abs(margin):.1f} punts")
        else:
            print(f"   Partit molt igualat")
        
        print(f"   Desviació estàndard: ±{stats['std_margin']:.1f} punts")
        
        # Total de punts
        print(f"\n🔢 TOTAL DE PUNTS:\n")
        
        exp_total = stats['expected_total']
        ci_total = stats['ci_95_total']
        
        print(f"   Total esperat: {exp_total:.1f} punts")
        print(f"   Interval 95%: [{ci_total[0]:.1f}, {ci_total[1]:.1f}]")
        print(f"   Desviació estàndard: ±{stats['std_total']:.1f}")
        
        # Over/Under
        print(f"\n   Over/Under 220 punts:")
        print(f"      Over (>220):  {stats['prob_over_220']*100:5.1f}%")
        print(f"      Under (≤220): {stats['prob_under_220']*100:5.1f}%")
        
        # NUEVO: Possessions i Pace
        print(f"\n⚡ POSSESSIONS I PACE:\n")
        
        exp_poss_a = stats['expected_possessions_a']
        exp_poss_b = stats['expected_possessions_b']
        exp_pace = stats['expected_pace']
        ci_pace = stats['ci_95_pace']
        
        print(f"   {stats['team_a']:25s} {exp_poss_a:5.1f} ± {stats['std_possessions_a']:4.1f} possessions")
        print(f"   {stats['team_b']:25s} {exp_poss_b:5.1f} ± {stats['std_possessions_b']:4.1f} possessions")
        
        print(f"\n   Pace esperat: {exp_pace:.1f} poss/equip")
        print(f"   Interval 95%: [{ci_pace[0]:.1f}, {ci_pace[1]:.1f}]")
        print(f"   Desviació estàndard: ±{stats['std_pace']:.1f}")
        
        print(f"\n{'='*70}")
    
    def analyze_distributions(self):
        """Analitza i mostra distribucions de puntuació."""
        
        if not self.simulation_results:
            print("⚠️  Cal executar run_monte_carlo() primer")
            return
        
        stats = self._calculate_statistics()
        
        print(f"\n{'='*70}")
        print(f"📊 ANÀLISI DE DISTRIBUCIONS")
        print(f"{'='*70}\n")
        
        # Distribució de puntuacions Team A
        scores_a = stats['score_distribution_a']
        
        print(f"🏀 {stats['team_a']}:\n")
        print(f"   Mínim: {np.min(scores_a):.0f} punts")
        print(f"   Q1 (25%): {np.percentile(scores_a, 25):.0f} punts")
        print(f"   Mediana: {np.median(scores_a):.0f} punts")
        print(f"   Mitjana: {np.mean(scores_a):.1f} punts")
        print(f"   Q3 (75%): {np.percentile(scores_a, 75):.0f} punts")
        print(f"   Màxim: {np.max(scores_a):.0f} punts")
        
        # Distribució de puntuacions Team B
        scores_b = stats['score_distribution_b']
        
        print(f"\n🏀 {stats['team_b']}:\n")
        print(f"   Mínim: {np.min(scores_b):.0f} punts")
        print(f"   Q1 (25%): {np.percentile(scores_b, 25):.0f} punts")
        print(f"   Mediana: {np.median(scores_b):.0f} punts")
        print(f"   Mitjana: {np.mean(scores_b):.1f} punts")
        print(f"   Q3 (75%): {np.percentile(scores_b, 75):.0f} punts")
        print(f"   Màxim: {np.max(scores_b):.0f} punts")
        
        # Histogram simple (ASCII)
        print(f"\n📊 Histograma de marcadors:\n")
        
        bins_a = np.histogram(scores_a, bins=10)
        bins_b = np.histogram(scores_b, bins=10)
        
        print(f"   {stats['team_a'][:20]:20s} Freqüència")
        for i in range(10):
            range_str = f"{bins_a[1][i]:.0f}-{bins_a[1][i+1]:.0f}"
            bar = '█' * int(bins_a[0][i] / max(bins_a[0]) * 30)
            print(f"   {range_str:10s} {bar} ({bins_a[0][i]})")
        
        print(f"\n   {stats['team_b'][:20]:20s} Freqüència")
        for i in range(10):
            range_str = f"{bins_b[1][i]:.0f}-{bins_b[1][i+1]:.0f}"
            bar = '█' * int(bins_b[0][i] / max(bins_b[0]) * 30)
            print(f"   {range_str:10s} {bar} ({bins_b[0][i]})")
        
        print(f"\n{'='*70}")
    
    def save_predictions(self, stats: Dict, filename: str = 'game_prediction.json'):
        """Guarda prediccions en JSON."""
        
        # Preparar per serialització JSON
        output = {
            'n_simulations': stats['n_simulations'],
            'team_a': stats['team_a'],
            'team_b': stats['team_b'],
            'probabilities': {
                'win_a': float(stats['prob_win_a']),
                'win_b': float(stats['prob_win_b']),
                'tie': float(stats['prob_tie'])
            },
            'expected_scores': {
                'team_a': float(stats['expected_score_a']),
                'team_b': float(stats['expected_score_b']),
                'margin_a': float(stats['expected_margin_a'])
            },
            'confidence_intervals': {
                'score_a_95': [float(stats['ci_95_score_a'][0]), float(stats['ci_95_score_a'][1])],
                'score_b_95': [float(stats['ci_95_score_b'][0]), float(stats['ci_95_score_b'][1])],
                'total_95': [float(stats['ci_95_total'][0]), float(stats['ci_95_total'][1])]
            },
            'totals': {
                'expected': float(stats['expected_total']),
                'prob_over_220': float(stats['prob_over_220']),
                'prob_under_220': float(stats['prob_under_220'])
            },
            'possessions': {
                'team_a': float(stats['expected_possessions_a']),
                'team_b': float(stats['expected_possessions_b']),
                'pace': float(stats['expected_pace'])
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Predicció guardada: {filename}")
    
    def plot_results(self, stats: Dict, save_path: str = 'monte_carlo_results.png'):
        """
        Crea visualitzacions dels resultats de Monte Carlo.
        
        Args:
            stats: Estadístiques calculades
            save_path: On guardar la imatge
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("\n⚠️  matplotlib no instal·lat")
            print("   Executa: pip install matplotlib")
            return
        
        # Crear figura amb 6 subplots
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'Predicció Monte Carlo: {stats["team_a"]} vs {stats["team_b"]}\n'
                    f'{stats["n_simulations"]:,} Simulacions', 
                    fontsize=16, fontweight='bold')
        
        # 1. Probabilitats de victòria (pie chart)
        ax1 = plt.subplot(2, 3, 1)
        sizes = [stats['prob_win_a'], stats['prob_win_b']]
        colors = ['#1f77b4', '#ff7f0e']
        labels = [f"{stats['team_a'][:15]}\n{stats['prob_win_a']*100:.1f}%",
                 f"{stats['team_b'][:15]}\n{stats['prob_win_b']*100:.1f}%"]
        ax1.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
        ax1.set_title('Probabilitat de Victòria', fontweight='bold')
        
        # 2. Distribució de puntuacions (histogrames)
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(stats['score_distribution_a'], bins=30, alpha=0.6, 
                label=stats['team_a'][:15], color='#1f77b4', edgecolor='black')
        ax2.hist(stats['score_distribution_b'], bins=30, alpha=0.6, 
                label=stats['team_b'][:15], color='#ff7f0e', edgecolor='black')
        ax2.axvline(stats['expected_score_a'], color='#1f77b4', 
                   linestyle='--', linewidth=2, label=f"Mitjana {stats['team_a'][:10]}")
        ax2.axvline(stats['expected_score_b'], color='#ff7f0e', 
                   linestyle='--', linewidth=2, label=f"Mitjana {stats['team_b'][:10]}")
        ax2.set_xlabel('Punts')
        ax2.set_ylabel('Freqüència')
        ax2.set_title('Distribució de Puntuacions', fontweight='bold')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribució de marge
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(stats['margin_distribution'], bins=40, alpha=0.7, 
                color='green', edgecolor='black')
        ax3.axvline(stats['expected_margin_a'], color='red', 
                   linestyle='--', linewidth=2, label=f"Marge esperat: {stats['expected_margin_a']:+.1f}")
        ax3.axvline(0, color='black', linestyle='-', linewidth=1)
        ax3.set_xlabel(f'Marge (+ = {stats["team_a"][:10]} guanya)')
        ax3.set_ylabel('Freqüència')
        ax3.set_title('Distribució del Marge de Victòria', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Possessions (box plot)
        ax4 = plt.subplot(2, 3, 4)
        poss_data = [stats['possessions_distribution_a'], 
                    stats['possessions_distribution_b']]
        bp = ax4.boxplot(poss_data, labels=[stats['team_a'][:15], stats['team_b'][:15]],
                        patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('#1f77b4')
        bp['boxes'][1].set_facecolor('#ff7f0e')
        ax4.set_ylabel('Possessions per partit')
        ax4.set_title('Distribució de Possessions', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Afegir mitjanes com text
        ax4.text(1, stats['expected_possessions_a'], 
                f"{stats['expected_possessions_a']:.1f}", 
                ha='center', va='bottom', fontweight='bold')
        ax4.text(2, stats['expected_possessions_b'], 
                f"{stats['expected_possessions_b']:.1f}", 
                ha='center', va='bottom', fontweight='bold')
        
        # 5. Total de punts i Over/Under
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(stats['total_distribution'], bins=40, alpha=0.7, 
                color='purple', edgecolor='black')
        ax5.axvline(stats['expected_total'], color='red', 
                   linestyle='--', linewidth=2, label=f"Total esperat: {stats['expected_total']:.1f}")
        ax5.axvline(220, color='orange', linestyle='--', linewidth=2, 
                   label=f"O/U 220: Over {stats['prob_over_220']*100:.1f}%")
        ax5.set_xlabel('Total de punts')
        ax5.set_ylabel('Freqüència')
        ax5.set_title('Distribució del Total de Punts', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Pace distribution
        ax6 = plt.subplot(2, 3, 6)
        ax6.hist(stats['pace_distribution'], bins=30, alpha=0.7, 
                color='teal', edgecolor='black')
        ax6.axvline(stats['expected_pace'], color='red', 
                   linestyle='--', linewidth=2, label=f"Pace esperat: {stats['expected_pace']:.1f}")
        ax6.axvline(100, color='gray', linestyle=':', linewidth=1, label='NBA mitjana: 100')
        ax6.set_xlabel('Pace (poss/equip)')
        ax6.set_ylabel('Freqüència')
        ax6.set_title('Distribució del Pace', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Gràfics guardats: {save_path}")
        
        try:
            plt.show()
        except:
            print("   (No es pot mostrar la finestra - gràfics guardats a arxiu)")
    
    def create_summary_visualization(self, stats: Dict, save_path: str = 'prediction_summary.png'):
        """
        Crea una visualització resumida d'una pàgina.
        
        Args:
            stats: Estadístiques calculades
            save_path: On guardar la imatge
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("\n⚠️  matplotlib no instal·lat")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{stats["team_a"]} vs {stats["team_b"]}\n'
                    f'Monte Carlo Prediction ({stats["n_simulations"]:,} simulacions)', 
                    fontsize=16, fontweight='bold')
        
        # 1. Win probability
        ax1.barh([0, 1], [stats['prob_win_a']*100, stats['prob_win_b']*100], 
                color=['#1f77b4', '#ff7f0e'])
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels([stats['team_a'][:20], stats['team_b'][:20]])
        ax1.set_xlabel('Probabilitat de Victòria (%)')
        ax1.set_title('Probabilitats de Victòria', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        for i, v in enumerate([stats['prob_win_a']*100, stats['prob_win_b']*100]):
            ax1.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
        
        # 2. Expected scores
        scores = [stats['expected_score_a'], stats['expected_score_b']]
        errors = [stats['std_score_a'], stats['std_score_b']]
        ax2.bar([0, 1], scores, yerr=errors, color=['#1f77b4', '#ff7f0e'], 
               capsize=10, alpha=0.7, edgecolor='black')
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels([stats['team_a'][:20], stats['team_b'][:20]])
        ax2.set_ylabel('Punts esperats')
        ax2.set_title('Marcador Esperat ± Desviació Estàndard', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, (s, e) in enumerate(zip(scores, errors)):
            ax2.text(i, s + e + 2, f'{s:.1f}±{e:.1f}', ha='center', fontweight='bold')
        
        # 3. Possessions
        poss = [stats['expected_possessions_a'], stats['expected_possessions_b']]
        poss_err = [stats['std_possessions_a'], stats['std_possessions_b']]
        ax3.bar([0, 1], poss, yerr=poss_err, color=['#1f77b4', '#ff7f0e'], 
               capsize=10, alpha=0.7, edgecolor='black')
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels([stats['team_a'][:20], stats['team_b'][:20]])
        ax3.set_ylabel('Possessions')
        ax3.set_title(f'Possessions Esperades (Pace: {stats["expected_pace"]:.1f})', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for i, (p, e) in enumerate(zip(poss, poss_err)):
            ax3.text(i, p + e + 0.5, f'{p:.1f}±{e:.1f}', ha='center', fontweight='bold')
        
        # 4. Summary table
        ax4.axis('off')
        summary_data = [
            ['Predicció Final', f'{stats["expected_score_a"]:.0f} - {stats["expected_score_b"]:.0f}'],
            ['Favorit', f'{stats["team_a"] if stats["prob_win_a"] > stats["prob_win_b"] else stats["team_b"]} '
                       f'({max(stats["prob_win_a"], stats["prob_win_b"])*100:.1f}%)'],
            ['Marge Esperat', f'{abs(stats["expected_margin_a"]):.1f} pts'],
            ['Total Punts', f'{stats["expected_total"]:.1f} pts'],
            ['Over 220', f'{stats["prob_over_220"]*100:.1f}%'],
            ['Under 220', f'{stats["prob_under_220"]*100:.1f}%'],
            ['Pace Mitjà', f'{stats["expected_pace"]:.1f} poss/equip'],
            ['Simulations', f'{stats["n_simulations"]:,}']
        ]
        
        table = ax4.table(cellText=summary_data, 
                         colLabels=['Mètrica', 'Valor'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Styling
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax4.set_title('Resum de la Predicció', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Resum visual guardat: {save_path}")
        
        try:
            plt.show()
        except:
            print("   (Gràfics guardats a arxiu)")



# ============================================================================
# EXEMPLE D'ÚS
# ============================================================================

def main():
    """Exemple complet de predicció de partit."""
    
    print("\n" + "="*70)
    print("🎲 PREDICCIÓ DE PARTIT AMB MONTE CARLO")
    print("="*70)
    
    # 1. Carregar equips
    print("\n📊 Carregant equips...\n")
    
    try:
        lakers = load_team_rotation("Lakers", max_lineups=20)
        warriors = load_team_rotation("Warriors", max_lineups=20)
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # 2. Crear predictor
    predictor = MonteCarloGamePredictor(lakers, warriors)
    
    # 3. Executar Monte Carlo
    stats = predictor.run_monte_carlo(n_simulations=1000, verbose=True)
    
    # 4. Mostrar predicció
    predictor.print_prediction(stats)
    
    # 5. Analitzar distribucions
    predictor.analyze_distributions()
    
    # 6. Crear visualitzacions
    print("\n" + "="*70)
    print("📊 GENERANT VISUALITZACIONS")
    print("="*70 + "\n")
    
    # Gràfics detallats
    predictor.plot_results(stats, save_path='lakers_vs_warriors_detailed.png')
    
    # Resum visual
    predictor.create_summary_visualization(stats, save_path='lakers_vs_warriors_summary.png')
    
    # 7. Guardar resultats JSON
    predictor.save_predictions(stats, 'lakers_vs_warriors_prediction.json')
    
    print("\n" + "="*70)
    print("✅ PREDICCIÓ COMPLETADA!")
    print("="*70)
    print("\n📁 Arxius generats:")
    print("   • lakers_vs_warriors_detailed.png (6 gràfics)")
    print("   • lakers_vs_warriors_summary.png (resum 1 pàgina)")
    print("   • lakers_vs_warriors_prediction.json (dades)")
    print("\n💡 Ara pots:")
    print("  • Veure els gràfics generats")
    print("  • Executar més simulacions per més precisió")
    print("  • Comparar diferents equips")
    print("  • Analitzar diferents escenaris")
    print()


if __name__ == "__main__":
    main()