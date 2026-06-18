"""
Aplicació Web Interactiva TFG - Sistema de Predicció NBA
========================================================

Aplicació Streamlit per:
1. Seleccionar equips i jugadors
2. Configurar minuts de joc
3. Simular partits en temps real
4. Visualitzar resultats amb gràfics interactius
5. Descarregar informes

Executa amb: streamlit run tfg_web_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import glob
import os
from realistic_game_simulator import RealisticGameSimulator, load_team_rotation
from monte_carlo_predictor import MonteCarloGamePredictor


# Configuració de la pàgina
st.set_page_config(
    page_title="TFG NBA Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalitzat - DISSENY MODERN MONOCROMÀTIC
st.markdown("""
<style>
    /* Imports de fonts modernes */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* Variables globals - paleta monocromàtica blau NBA */
    :root {
        --primary: #17408B;        /* NBA Blue - color principal */
        --primary-dark: #0F2D5F;
        --primary-light: #4A6FB5;
        --accent: #17408B;
        --success: #00C853;
        --danger: #D32F2F;
        --warning: #FF9800;
        --dark: #1A1A2E;
        --light: #F8F9FA;
        --gray: #6C757D;
        --shadow-sm: 0 2px 4px rgba(0,0,0,0.08);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.12);
        --shadow-lg: 0 8px 24px rgba(0,0,0,0.15);
    }
    
    /* Font global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Header principal - MONOCROMÀTIC */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        padding: 2rem 1rem;
        color: var(--primary);
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .subtitle {
        text-align: center;
        color: var(--gray);
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-out 0.3s both;
    }
    
    /* Animacions */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Targetes d'equip */
    .team-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
        border-left: 4px solid var(--primary);
    }
    
    .team-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
    }
    
    /* Caixes d'estadístiques */
    .stat-box {
        text-align: center;
        padding: 1.5rem;
        border-radius: 12px;
        background: white;
        margin: 0.5rem 0;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .stat-box:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary);
        line-height: 1.2;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: var(--gray);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background: white;
        border-radius: 12px;
        color: var(--gray);
        font-weight: 600;
        font-size: 0.95rem;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f8f9fa;
        color: var(--dark);
        border-color: var(--primary);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: white !important;
        border-color: transparent !important;
        box-shadow: var(--shadow-md);
    }
    
    /* Botons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
        border: none;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    .stButton > button[kind="primary"] {
        background: var(--primary) !important;
        color: white !important;
    }
    
    /* SIDEBAR BLAVA - només dins de la sidebar */
    [data-testid="stSidebar"] {
        background: var(--primary) !important;
        border-right: none;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: var(--primary) !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p {
        color: white !important;
    }
    
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCheckbox label > div,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* ============================================
       SLIDERS DE LA SIDEBAR - FORÇA EL COLOR BLANC
       ============================================ */
    
    /* Track del slider (la barra base) */
    section[data-testid="stSidebar"] div[data-baseweb="slider"] > div:first-child {
        background: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Tots els divs interiors del slider */
    section[data-testid="stSidebar"] div[data-baseweb="slider"] div {
        background-color: white !important;
    }
    
    /* El "thumb" (cercle que es mou) */
    section[data-testid="stSidebar"] div[role="slider"] {
        background: white !important;
        background-color: white !important;
        border: 2px solid white !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    
    /* Track base del slider - vermell -> blanc */
    section[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
        color: white !important;
        background: transparent !important;
    }
    
    /* Override agressiu de TOTS els colors de fons taronges/vermells */
    section[data-testid="stSidebar"] *[style*="background-color: rgb(255"],
    section[data-testid="stSidebar"] *[style*="background: rgb(255"] {
        background: white !important;
        background-color: white !important;
    }
    
    /* Force tots els elements del slider a blanc */
    section[data-testid="stSidebar"] .stSlider div[data-baseweb] * {
        background-color: white !important;
    }
    
    section[data-testid="stSidebar"] .stSlider div[data-baseweb] > div:first-child {
        background-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Text del slider (valors min/max) */
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] *,
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: white !important;
        background: transparent !important;
    }
    
    /* ============================================
       SELECTBOX - TEXT SEMPRE VISIBLE
       ============================================ */
    
    /* Container del selectbox al main */
    .main .stSelectbox {
        color: #1A1A2E !important;
    }
    
    /* Tot dins del selectbox del main */
    .main .stSelectbox * {
        color: #1A1A2E !important;
    }
    
    /* El text visible del selectbox seleccionat */
    .main div[data-baseweb="select"] {
        background-color: white !important;
    }
    
    .main div[data-baseweb="select"] > div {
        background-color: white !important;
        color: #1A1A2E !important;
    }
    
    .main div[data-baseweb="select"] > div > div {
        color: #1A1A2E !important;
    }
    
    /* Input del selectbox */
    .main div[data-baseweb="select"] input {
        color: #1A1A2E !important;
        background-color: white !important;
    }
    
    /* El text que es mostra (label seleccionat) */
    .main div[data-baseweb="select"] [data-testid="stMarkdownContainer"] {
        color: #1A1A2E !important;
    }
    
    /* Spans dins del selectbox */
    .main div[data-baseweb="select"] span {
        color: #1A1A2E !important;
    }
    
    /* SVG icon */
    .main div[data-baseweb="select"] svg {
        fill: #1A1A2E !important;
    }
    
    /* Override per qualsevol selectbox a tot el lloc */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background-color: white !important;
    }
    
    [data-testid="stSelectbox"] [data-baseweb="select"] > div > div:first-child {
        color: #1A1A2E !important;
    }
    
    /* Forçar opacity dels elements del selectbox */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div > div:first-child,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div > div:first-child * {
        opacity: 1 !important;
        color: #1A1A2E !important;
    }
    
    /* Expander dins la sidebar */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        border-color: white !important;
    }
    
    /* Checkbox/toggle dins la sidebar */
    [data-testid="stSidebar"] [data-baseweb="checkbox"] {
        color: white !important;
    }
    
    /* IMPORTANT: NO afectar elements de MAIN content */
    
    /* Mètriques de Streamlit */
    [data-testid="stMetric"] {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: var(--shadow-sm);
        border-left: 4px solid var(--primary);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--dark) !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 600 !important;
        color: var(--gray) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.8rem !important;
    }
    
    /* Selectbox - estil bàsic */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--primary);
    }
    
    /* Sliders al main */
    .stSlider > div > div > div > div {
        background: var(--primary) !important;
    }
    
    /* Alertes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: var(--shadow-sm);
        padding: 1rem 1.2rem;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: var(--primary);
        background: #f8f9fa;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    
    /* Score display - MONOCROMÀTIC */
    .score-display {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 2rem;
        padding: 2rem;
        background: white;
        border-radius: 20px;
        box-shadow: var(--shadow-lg);
        margin: 1.5rem 0;
        animation: fadeIn 0.5s ease-out;
    }
    
    .team-score {
        text-align: center;
        flex: 1;
    }
    
    .team-name-big {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--dark);
        margin-bottom: 0.5rem;
    }
    
    .score-number {
        font-size: 5rem;
        font-weight: 800;
        line-height: 1;
        color: var(--primary);
    }
    
    .score-number-loser {
        font-size: 5rem;
        font-weight: 800;
        line-height: 1;
        color: var(--gray);
    }
    
    .score-divider {
        font-size: 3rem;
        font-weight: 300;
        color: var(--gray);
    }
    
    .winner-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        background: var(--primary);
        color: white;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Win probability bar */
    .win-prob-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
        margin: 1rem 0;
    }
    
    .win-prob-bar {
        height: 40px;
        border-radius: 20px;
        overflow: hidden;
        display: flex;
        background: #f0f0f0;
        margin: 1rem 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .win-prob-team-a {
        background: var(--primary);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 0.95rem;
        transition: width 0.5s ease;
    }
    
    .win-prob-team-b {
        background: var(--primary-light);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 0.95rem;
        transition: width 0.5s ease;
    }
    
    /* Info cards */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: var(--shadow-sm);
        margin: 0.5rem 0;
        border-left: 3px solid var(--primary);
    }
    
    .info-card-title {
        font-size: 0.85rem;
        color: var(--gray);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .info-card-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--dark);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: var(--dark) !important;
        letter-spacing: -0.5px;
    }
    
    /* Eliminar padding excessiu */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: var(--gray);
        font-size: 0.85rem;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: var(--primary) !important;
    }
    
    /* Code blocks */
    code {
        background: #f8f9fa !important;
        color: var(--primary) !important;
        padding: 0.2rem 0.5rem !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9em !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: var(--primary) !important;
    }
    
    /* Hide Streamlit footer */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    
    /* Plotly charts container */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
        background: white;
        padding: 0.5rem;
    }
    
    /* ============================================
       SIDEBAR - TEXT BLANC, SLIDERS NETS
       ============================================ */
    
    /* Tots els textos de la sidebar en blanc */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown *,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6 {
        color: white !important;
    }
    
    /* Labels dels widgets (Simulacions Monte Carlo, etc) */
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] label p {
        color: white !important;
    }
    
    /* Label del checkbox/toggle */
    section[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stCheckbox label p {
        color: white !important;
    }
    
    /* ============================================
       SIDEBAR - SELECTBOX I NUMBER INPUT
       ============================================ */
    
    /* SELECTBOX a la sidebar */
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
        background-color: white !important;
        border: 2px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 8px !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
        background-color: white !important;
        color: #17408B !important;
        font-weight: 600 !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span {
        color: #17408B !important;
        font-weight: 600 !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] svg {
        fill: #17408B !important;
    }
    
    /* Hover del selectbox */
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"]:hover {
        border-color: white !important;
    }
    
    /* NUMBER INPUT a la sidebar */
    section[data-testid="stSidebar"] .stNumberInput input {
        background-color: white !important;
        color: #17408B !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        text-align: center !important;
        border: 2px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 8px !important;
    }
    
    section[data-testid="stSidebar"] .stNumberInput input:focus {
        border-color: white !important;
        box-shadow: 0 0 0 2px rgba(255,255,255,0.3) !important;
    }
    
    /* Botons +/- del number input */
    section[data-testid="stSidebar"] .stNumberInput button {
        background-color: white !important;
        color: #17408B !important;
        border: 2px solid rgba(255, 255, 255, 0.5) !important;
        font-weight: 700 !important;
    }
    
    section[data-testid="stSidebar"] .stNumberInput button:hover {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-color: white !important;
    }
    
    /* Captions */
    section[data-testid="stSidebar"] [class*="caption"] {
        color: rgba(255, 255, 255, 0.7) !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# FUNCIONS AUXILIARS PER IMATGES
# ============================================

# Mapa NOM EQUIP COMPLET → Abreviació (per buscar el fitxer)
TEAM_NAME_TO_ABBR = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS'
}


# ============================================
# COLORS OFICIALS DELS EQUIPS NBA
# ============================================
# Format: {'primary': color principal, 'secondary': color secundari}
TEAM_COLORS = {
    'ATL': {'primary': '#E03A3E', 'secondary': '#C1D32F'},
    'BOS': {'primary': '#007A33', 'secondary': '#BA9653'},
    'BKN': {'primary': '#000000', 'secondary': '#FFFFFF'},
    'CHA': {'primary': '#1D1160', 'secondary': '#00788C'},
    'CHI': {'primary': '#CE1141', 'secondary': '#000000'},
    'CLE': {'primary': '#860038', 'secondary': '#FDBB30'},
    'DAL': {'primary': '#00538C', 'secondary': '#002B5E'},
    'DEN': {'primary': '#0E2240', 'secondary': '#FEC524'},
    'DET': {'primary': '#C8102E', 'secondary': '#1D42BA'},
    'GSW': {'primary': '#1D428A', 'secondary': '#FFC72C'},
    'HOU': {'primary': '#CE1141', 'secondary': '#000000'},
    'IND': {'primary': '#002D62', 'secondary': '#FDBB30'},
    'LAC': {'primary': '#C8102E', 'secondary': '#1D428A'},
    'LAL': {'primary': '#552583', 'secondary': '#FDB927'},
    'MEM': {'primary': '#5D76A9', 'secondary': '#12173F'},
    'MIA': {'primary': '#98002E', 'secondary': '#F9A01B'},
    'MIL': {'primary': '#00471B', 'secondary': '#EEE1C6'},
    'MIN': {'primary': '#0C2340', 'secondary': '#236192'},
    'NOP': {'primary': '#0C2340', 'secondary': '#C8102E'},
    'NYK': {'primary': '#006BB6', 'secondary': '#F58426'},
    'OKC': {'primary': '#007AC1', 'secondary': '#EF3B24'},
    'ORL': {'primary': '#0077C0', 'secondary': '#C4CED4'},
    'PHI': {'primary': '#006BB6', 'secondary': '#ED174C'},
    'PHX': {'primary': '#1D1160', 'secondary': '#E56020'},
    'POR': {'primary': '#E03A3E', 'secondary': '#000000'},
    'SAC': {'primary': '#5A2D81', 'secondary': '#63727A'},
    'SAS': {'primary': '#C4CED4', 'secondary': '#000000'},
    'TOR': {'primary': '#CE1141', 'secondary': '#000000'},
    'UTA': {'primary': '#002B5C', 'secondary': '#00471B'},
    'WAS': {'primary': '#002B5C', 'secondary': '#E31837'},
}


def get_team_colors(team_name):
    """
    Retorna els colors primari i secundari d'un equip.
    
    Args:
        team_name: Nom de l'equip (complet o abreviació)
    
    Returns:
        Dict amb 'primary' i 'secondary'
    """
    # Si ja és una abreviació
    if len(team_name) == 3 and team_name.isupper():
        abbr = team_name
    else:
        abbr = TEAM_NAME_TO_ABBR.get(team_name)
    
    return TEAM_COLORS.get(abbr, {'primary': '#17408B', 'secondary': '#C9082A'})


def get_team_primary_color(team_name):
    """Retorna el color primari d'un equip"""
    return get_team_colors(team_name)['primary']


def get_team_secondary_color(team_name):
    """Retorna el color secundari d'un equip"""
    return get_team_colors(team_name)['secondary']


def get_team_logo_path(team_name):
    """
    Retorna el path de l'escut de l'equip si existeix, sinó None.
    
    Accepta tant noms complets ('Los Angeles Lakers') com abreviacions ('LAL').
    Prioritza SVG sobre PNG perquè es veu millor.
    """
    import os
    
    # Si ja és una abreviació
    if len(team_name) == 3 and team_name.isupper():
        abbr = team_name
    else:
        # Convertir nom complet a abreviació
        abbr = TEAM_NAME_TO_ABBR.get(team_name)
        if not abbr:
            # Provar amb el nom tal qual
            abbr = team_name
    
    # Provar diferents formats (SVG primer per millor qualitat!)
    for ext in ['svg', 'png', 'webp', 'jpg']:
        path = f'imatges/escuts/{abbr}.{ext}'
        if os.path.exists(path):
            return path
    
    return None


def get_team_logo_html(team_name, size=40):
    """
    Retorna HTML amb l'escut de l'equip, o un placeholder si no existeix.
    Tracta els SVG codificant-los a base64 per evitar interferències del CSS.
    IMPORTANT: Els colors dels escuts NO es modifiquen mai.
    """
    import base64
    
    path = get_team_logo_path(team_name)
    
    if path:
        try:
            ext = path.split('.')[-1].lower()
            
            # Llegir el fitxer
            with open(path, 'rb') as f:
                file_content = f.read()
            
            # Codificar a base64
            img_b64 = base64.b64encode(file_content).decode()
            
            # Determinar MIME type
            if ext == 'svg':
                mime = 'svg+xml'
            elif ext == 'jpg' or ext == 'jpeg':
                mime = 'jpeg'
            else:
                mime = ext
            
            # Carregar com a <img> (el CSS de la pàgina NO pot modificar-lo)
            return f'<img src="data:image/{mime};base64,{img_b64}" style="width: {size}px; height: {size}px; object-fit: contain; vertical-align: middle; flex-shrink: 0;" alt="{team_name}"/>'
            
        except Exception as e:
            pass
    
    # Placeholder amb emoji
    return f'<span style="font-size: {size}px;">🏀</span>'


def get_player_photo_path(player_name):
    """
    Retorna el path de la foto del jugador si existeix, sinó None.
    
    Format: 'LeBron James' → 'imatges/jugadors/LeBron_James.png'
    """
    import os
    
    # Convertir nom a format de fitxer
    safe_name = player_name.replace(' ', '_')
    safe_name = safe_name.replace('.', '')
    safe_name = safe_name.replace("'", '')
    safe_name = safe_name.replace('"', '')
    
    # Provar diferents formats
    for ext in ['png', 'jpg', 'jpeg', 'webp']:
        path = f'imatges/jugadors/{safe_name}.{ext}'
        if os.path.exists(path):
            return path
    
    return None


def get_player_photo_html(player_name, size=50, round_img=True):
    """
    Retorna HTML amb la foto del jugador, o un placeholder si no existeix.
    """
    import base64
    
    path = get_player_photo_path(player_name)
    
    if path:
        try:
            with open(path, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode()
            
            ext = path.split('.')[-1].lower()
            border_radius = '50%' if round_img else '8px'
            
            return f'<img src="data:image/{ext};base64,{img_b64}" style="width: {size}px; height: {size}px; object-fit: cover; border-radius: {border_radius}; vertical-align: middle; border: 2px solid #17408B;" alt="{player_name}"/>'
        except:
            pass
    
    # Placeholder amb emoji
    return f'<span style="font-size: {size}px;">👤</span>'


# ============================================
# FUNCIONS DE CÀRREGA DE DADES
# ============================================


@st.cache_data
def load_data():
    """Carrega dades dels CSV amb cerca flexible."""
    import os
    import glob
    
    try:
        # Buscar arxius CSV de jugadors
        player_csv_paths = [
            'nba_data_advanced/advanced_player_data.csv',
            'nba_data_advanced/nba_player_data.csv',
            'advanced_player_data.csv',
            'nba_player_data.csv'
        ]
        
        players_df = None
        for path in player_csv_paths:
            if os.path.exists(path):
                players_df = pd.read_csv(path)
                st.sidebar.success(f"✅ Jugadors carregats de: {path}")
                break
        
        # Si no troba cap, buscar qualsevol CSV a nba_data_advanced
        if players_df is None:
            pattern = 'nba_data_advanced/*.csv'
            csv_files = glob.glob(pattern)
            if csv_files:
                players_df = pd.read_csv(csv_files[0])
                st.sidebar.success(f"✅ Jugadors carregats de: {csv_files[0]}")
            else:
                st.error("❌ No s'ha trobat cap CSV de jugadors a nba_data_advanced/")
                return None, None, None
        
        # Buscar arxius CSV de lineups
        lineup_csv_paths = [
            'nba_lineups/nba_lineups.csv',
            'nba_lineups/lineups_data.csv',
            'nba_lineups.csv',
            'lineups_data.csv'
        ]
        
        lineups_df = None
        for path in lineup_csv_paths:
            if os.path.exists(path):
                lineups_df = pd.read_csv(path)
                st.sidebar.success(f"✅ Lineups carregats de: {path}")
                break
        
        # Si no troba cap, buscar qualsevol CSV a nba_lineups
        if lineups_df is None:
            pattern = 'nba_lineups/*.csv'
            csv_files = glob.glob(pattern)
            if csv_files:
                lineups_df = pd.read_csv(csv_files[0])
                st.sidebar.success(f"✅ Lineups carregats de: {csv_files[0]}")
            else:
                st.error("❌ No s'ha trobat cap CSV de lineups a nba_lineups/")
                return None, None, None
        
        # Mapping d'abreviatures a noms complets
        TEAM_ABBREV_TO_FULL = {
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
        
        # Si team_name és abreviatura, convertir a nom complet
        if 'team_name' in players_df.columns:
            # Expandir abreviatures
            players_df['team_full_name'] = players_df['team_name'].map(TEAM_ABBREV_TO_FULL)
            
            # Si no troba mapping, mantenir original
            players_df['team_full_name'] = players_df['team_full_name'].fillna(players_df['team_name'])
            
            st.sidebar.success(f"✅ {len(players_df)} jugadors carregats")
        else:
            # Enriquir des de lineups
            player_team_map = {}
            
            for _, lineup in lineups_df.iterrows():
                team_name = lineup['TEAM_NAME']
                player_names_str = lineup.get('PLAYER_NAMES', '')
                
                if pd.notna(player_names_str) and isinstance(player_names_str, str):
                    player_names = player_names_str.split(' - ')
                    
                    for player_name in player_names:
                        player_name = player_name.strip()
                        if player_name and player_name not in player_team_map:
                            player_team_map[player_name] = team_name
            
            players_df['team_full_name'] = players_df['player_name'].map(player_team_map)
            players_df = players_df[players_df['team_full_name'].notna()].copy()
            
            st.sidebar.success(f"✅ {len(players_df)} jugadors amb equip")
        
        teams = sorted(lineups_df['TEAM_NAME'].unique())
        
        return players_df, lineups_df, teams
    except Exception as e:
        st.error(f"Error carregant dades: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None


def init_session_state():
    """Inicialitza l'estat de la sessió."""
    if 'team_a_players' not in st.session_state:
        st.session_state.team_a_players = {}
    if 'team_b_players' not in st.session_state:
        st.session_state.team_b_players = {}
    if 'simulation_result' not in st.session_state:
        st.session_state.simulation_result = None
    if 'mc_stats' not in st.session_state:
        st.session_state.mc_stats = None


def render_player_selector(team_name, players_df, key_prefix):
    """
    Renderitza selector de jugadors amb sliders.
    
    Args:
        team_name: Nom de l'equip
        players_df: DataFrame amb jugadors
        key_prefix: Prefix per keys únic ('a' o 'b')
    
    Returns:
        Dict amb {player_name: minutes}
    """
    # Detectar noms de columnes (poden variar)
    # Buscar columna de nom de jugador
    name_col = None
    for col in ['player_name', 'PLAYER', 'player', 'Player', 'PLAYER_NAME', 'name']:
        if col in players_df.columns:
            name_col = col
            break
    
    if name_col is None:
        st.error("No s'ha trobat la columna de nom al CSV")
        return {}
    
    # Buscar columna de minuts
    min_col = None
    for col in ['MPG', 'min', 'MIN', 'minutes', 'Minutes', 'MINUTES', 'mpg']:
        if col in players_df.columns:
            min_col = col
            break
    
    if min_col is None:
        st.error("No s'ha trobat la columna de minuts al CSV")
        return {}
    
    # Buscar columna de punts
    pts_col = None
    for col in ['PTS', 'pts', 'points', 'Points', 'POINTS']:
        if col in players_df.columns:
            pts_col = col
            break
    
    # Buscar columna de partits jugats
    gp_col = None
    for col in ['GP', 'gp', 'games', 'Games', 'GAMES']:
        if col in players_df.columns:
            gp_col = col
            break
    
    # Si no hi ha columna d'equip, filtrar per tots els jugadors
    # (assumim que l'usuari ja ha filtrat el CSV o que tots són del mateix equip)
    team_col = None
    for col in ['team_full_name', 'team_name', 'TEAM', 'team', 'Team', 'TEAM_NAME']:
        if col in players_df.columns:
            team_col = col
            break
    
    if team_col:
        # Filtrar jugadors de l'equip (matching exacte o parcial)
        team_players = players_df[
            players_df[team_col].str.contains(team_name, case=False, na=False)
        ].copy()
    else:
        st.error(f"❌ No s'ha trobat columna d'equip al CSV")
        st.info(f"Columnes disponibles: {list(players_df.columns[:10])}")
        return {}
    
    if len(team_players) == 0:
        st.error(f"❌ No hi ha jugadors disponibles per **{team_name}**")
        
        # DEBUG INFO
        with st.expander("🔍 Debug Info"):
            st.write(f"**Equip buscat:** {team_name}")
            st.write(f"**Total jugadors al CSV:** {len(players_df)}")
            
            if team_col and team_col in players_df.columns:
                st.write(f"**Columna equip:** {team_col}")
                unique_teams = players_df[team_col].unique()
                st.write(f"**Equips disponibles ({len(unique_teams)}):**")
                st.write(sorted([str(t) for t in unique_teams if pd.notna(t)])[:20])
            else:
                st.write(f"**Columnes disponibles:** {list(players_df.columns[:10])}")
        
        st.warning("💡 **Solucions:**\n"
                  "1. Verifica que has executat el scraper: `python final_data_scraper.py --test`\n"
                  "2. Comprova que el CSV `advanced_player_data.csv` té la columna `team_name`\n"
                  "3. Prova amb un equip diferent")
        
        return {}
    
    # Ordenar per minuts
    team_players = team_players.sort_values(min_col, ascending=False)
    
    # Mostrar top 16 jugadors
    selected_players = {}
    
    # Capçalera amb escut de l'equip + color de l'equip
    team_logo_html = get_team_logo_html(team_name, size=35)
    team_color = get_team_primary_color(team_name)
    st.caption(f"Mostrant top 16 jugadors per minuts ({min_col})")
    
    # Selecciona tots / cap
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Selecciona Tots", key=f"{key_prefix}_select_all"):
            # Marcar per seleccionar tots al següent rerun
            st.session_state[f'{key_prefix}_select_all_flag'] = True
            st.rerun()
    with col2:
        if st.button("❌ Deselecciona Tots", key=f"{key_prefix}_deselect_all"):
            # Marcar per deseleccionar tots al següent rerun
            st.session_state[f'{key_prefix}_deselect_all_flag'] = True
            st.rerun()
    
    st.markdown("---")
    
    # Comprovar flags ABANS de crear widgets
    select_all_active = st.session_state.get(f'{key_prefix}_select_all_flag', False)
    deselect_all_active = st.session_state.get(f'{key_prefix}_deselect_all_flag', False)
    
    for idx, (_, player) in enumerate(team_players.head(16).iterrows()):
        player_name = player[name_col]
        
        # MPG està en minuts per partit, així que multiplicar per 1 per partit de 48 min
        default_mins = min(int(player[min_col]), 40)  # MPG normalment és <40
        
        # Determinar si està actiu
        if select_all_active:
            default_active = True
        elif deselect_all_active:
            default_active = False
        else:
            default_active = idx < 8  # Top 8 actius per defecte
        
        with st.container():
            col_photo, col1, col2 = st.columns([0.6, 2.4, 2])
            
            with col_photo:
                # Foto del jugador
                player_photo = get_player_photo_html(player_name, size=40, round_img=True)
                st.markdown(f"<div style='padding-top: 0.3rem;'>{player_photo}</div>", unsafe_allow_html=True)
            
            with col1:
                # Checkbox + nom
                is_active = st.checkbox(
                    f"**{player_name}**",
                    value=default_active,
                    key=f"{key_prefix}_check_{idx}"
                )
            
            with col2:
                if is_active:
                    # Slider minuts
                    minutes = st.slider(
                        "Minuts",
                        min_value=0,
                        max_value=48,
                        value=default_mins,
                        key=f"{key_prefix}_mins_{idx}",
                        label_visibility="collapsed"
                    )
                    selected_players[player_name] = minutes
                    
                    # Mostrar info
                    info_parts = []
                    if min_col and pd.notna(player[min_col]):
                        info_parts.append(f"{player[min_col]:.1f} mpg")
                    if pts_col and pd.notna(player[pts_col]):
                        info_parts.append(f"{player[pts_col]:.1f} pts")
                    if gp_col and pd.notna(player[gp_col]):
                        info_parts.append(f"{int(player[gp_col])} GP")
                    
                    if info_parts:
                        st.caption(f"📊 Real: {' | '.join(info_parts)}")
    
    # Reset flags després de processar
    if select_all_active:
        st.session_state[f'{key_prefix}_select_all_flag'] = False
    if deselect_all_active:
        st.session_state[f'{key_prefix}_deselect_all_flag'] = False
    
    return selected_players


def display_team_summary(team_name, players_dict):
    """Mostra resum de l'equip seleccionat."""
    if not players_dict:
        st.warning("Cap jugador seleccionat")
        return
    
    total_mins = sum(players_dict.values())
    n_players = len(players_dict)
    
    # Validació i ajust automàtic a 240 minuts
    target_mins = 240
    difference = total_mins - target_mins
    
    # Mètriques
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{n_players}</div>
            <div class="stat-label">Jugadors</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "#4CAF50" if abs(difference) < 5 else "#ff7f0e"
        st.markdown(f"""
        <div class="stat-box" style="background-color: {color}20;">
            <div class="stat-value" style="color: {color};">{total_mins}</div>
            <div class="stat-label">Minuts Totals (objectiu: 240)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_mins = total_mins / n_players if n_players > 0 else 0
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{avg_mins:.1f}</div>
            <div class="stat-label">Mitjana Min/Jug</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Advertència si no suma 240
    if abs(difference) > 5:
        st.warning(f"⚠️ Els minuts totals són {total_mins}, però haurien de sumar 240 per cobrir 5 jugadors × 48 minuts")
        st.info(f"💡 Ajusta els minuts: {'Redueix' if difference > 0 else 'Augmenta'} {abs(difference)} minuts en total")
    elif abs(difference) > 0:
        st.success(f"✅ Minuts totals: {total_mins} (diferència: {difference:+d} min)")
    else:
        st.success(f"✅ Perfecte! Exactament 240 minuts")
    
    # Llista de jugadors
    st.markdown("#### 📋 Rotació:")
    sorted_players = sorted(players_dict.items(), key=lambda x: x[1], reverse=True)
    
    for i, (player, mins) in enumerate(sorted_players):
        progress = mins / 48
        color = "#4CAF50" if mins > 0 else "#666"
        st.progress(progress, text=f"{i+1}. **{player[:25]}** - {mins} min")


def plot_win_probability(stats):
    """Gràfic de probabilitat de victòria (Plotly) amb colors dels equips."""
    # Obtenir colors dels equips
    color_a = get_team_primary_color(stats['team_a'])
    color_b = get_team_primary_color(stats['team_b'])
    
    # Si els colors són massa similars, usar secundari del B
    if color_a.lower() == color_b.lower():
        color_b = get_team_secondary_color(stats['team_b'])
    
    fig = go.Figure(data=[go.Pie(
        labels=[stats['team_a'][:20], stats['team_b'][:20]],
        values=[stats['prob_win_a'], stats['prob_win_b']],
        hole=0.4,
        marker_colors=[color_a, color_b],
        textinfo='label+percent',
        textfont_size=14
    )])
    
    fig.update_layout(
        title="Probabilitat de Victòria",
        height=400,
        showlegend=False,
        font=dict(family="Inter, sans-serif")
    )
    
    return fig


def plot_score_distributions(stats):
    """Gràfic de distribucions de puntuacions amb colors dels equips."""
    color_a = get_team_primary_color(stats['team_a'])
    color_b = get_team_primary_color(stats['team_b'])
    
    if color_a.lower() == color_b.lower():
        color_b = get_team_secondary_color(stats['team_b'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=stats['score_distribution_a'],
        name=stats['team_a'][:20],
        opacity=0.7,
        marker_color=color_a,
        nbinsx=30
    ))
    
    fig.add_trace(go.Histogram(
        x=stats['score_distribution_b'],
        name=stats['team_b'][:20],
        opacity=0.7,
        marker_color=color_b,
        nbinsx=30
    ))
    
    fig.add_vline(x=stats['expected_score_a'], line_dash="dash", 
                  line_color=color_a, annotation_text=f"Mitjana A: {stats['expected_score_a']:.1f}")
    fig.add_vline(x=stats['expected_score_b'], line_dash="dash", 
                  line_color=color_b, annotation_text=f"Mitjana B: {stats['expected_score_b']:.1f}")
    
    fig.update_layout(
        title="Distribució de Puntuacions",
        xaxis_title="Punts",
        yaxis_title="Freqüència",
        barmode='overlay',
        height=400,
        font=dict(family="Inter, sans-serif")
    )
    
    return fig


def plot_total_points(stats):
    """Gràfic de total de punts amb color de l'equip local."""
    color_a = get_team_primary_color(stats['team_a'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=stats['total_distribution'],
        marker_color=color_a,
        opacity=0.7,
        nbinsx=40
    ))
    
    fig.add_vline(x=stats['expected_total'], line_dash="dash", 
                  line_color=color_a, annotation_text=f"Mitjana: {stats['expected_total']:.1f}")
    fig.add_vline(x=220, line_dash="dash", 
                  line_color="#6C757D", annotation_text=f"O/U 220")
    
    fig.update_layout(
        title=f"Total de Punts (Over 220: {stats['prob_over_220']*100:.1f}%)",
        xaxis_title="Total Punts",
        yaxis_title="Freqüència",
        height=400,
        font=dict(family="Inter, sans-serif")
    )
    
    return fig


def plot_margin_distribution(stats):
    """Distribució del marge de victòria amb color de l'equip local."""
    import numpy as np
    color_a = get_team_primary_color(stats['team_a'])
    
    fig = go.Figure()
    
    # Calcular marges (positiu = guanya equip A)
    scores_a = stats['score_distribution_a']
    scores_b = stats['score_distribution_b']
    margins = [a - b for a, b in zip(scores_a, scores_b)]
    
    # Histograma de marges
    fig.add_trace(go.Histogram(
        x=margins,
        marker_color=color_a,
        opacity=0.75,
        nbinsx=30,
        hovertemplate='Marge: <b>%{x}</b><br>Freqüència: %{y}<extra></extra>'
    ))
    
    # Línia vertical al zero (empat)
    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color="#D32F2F",
        line_width=2,
        annotation_text="Empat",
        annotation_position="top"
    )
    
    # Mitjana del marge
    mean_margin = np.mean(margins)
    fig.add_vline(
        x=mean_margin,
        line_dash="dash",
        line_color="#00C853",
        line_width=2,
        annotation_text=f"Mitjana: {mean_margin:+.1f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title=f"Marge de Victòria",
        xaxis_title=f"Marge (+ favorable a {stats['team_a'][:15]} | - favorable a {stats['team_b'][:15]})",
        yaxis_title="Freqüència",
        height=400,
        font=dict(family="Inter, sans-serif"),
        showlegend=False
    )
    
    return fig


def plot_offensive_rating(stats):
    """Comparació d'Offensive Rating dels dos equips amb colors dels equips."""
    import numpy as np
    
    color_a = get_team_primary_color(stats['team_a'])
    color_b = get_team_primary_color(stats['team_b'])
    
    if color_a.lower() == color_b.lower():
        color_b = get_team_secondary_color(stats['team_b'])
    
    # Calcular ortg estimat: punts per 100 possessions
    scores_a = np.array(stats['score_distribution_a'])
    scores_b = np.array(stats['score_distribution_b'])
    poss_a = np.array(stats['possessions_distribution_a'])
    poss_b = np.array(stats['possessions_distribution_b'])
    
    ortg_a = (scores_a / poss_a) * 100
    ortg_b = (scores_b / poss_b) * 100
    
    fig = go.Figure()
    
    # Convertir hex a rgba per fillcolor amb transparència
    def hex_to_rgba(hex_color, alpha=0.3):
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r}, {g}, {b}, {alpha})'
    
    fig.add_trace(go.Box(
        y=ortg_a,
        name=stats['team_a'][:20],
        marker_color=color_a,
        fillcolor=hex_to_rgba(color_a, 0.3),
        line_color=color_a,
        boxmean='sd',
        hovertemplate='<b>%{x}</b><br>ORtg: %{y:.1f}<extra></extra>'
    ))
    
    fig.add_trace(go.Box(
        y=ortg_b,
        name=stats['team_b'][:20],
        marker_color=color_b,
        fillcolor=hex_to_rgba(color_b, 0.3),
        line_color=color_b,
        boxmean='sd',
        hovertemplate='<b>%{x}</b><br>ORtg: %{y:.1f}<extra></extra>'
    ))
    
    fig.add_hline(
        y=115,
        line_dash="dot",
        line_color="#6C757D",
        line_width=1,
        annotation_text="Mitjana NBA: 115",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Offensive Rating (punts per 100 possessions)",
        yaxis_title="Offensive Rating",
        height=400,
        font=dict(family="Inter, sans-serif"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_win_scenarios(stats):
    """Gràfic de barres mostrant escenaris de victòria per franges amb colors d'equips."""
    import numpy as np
    
    color_a = get_team_primary_color(stats['team_a'])
    color_b = get_team_primary_color(stats['team_b'])
    
    if color_a.lower() == color_b.lower():
        color_b = get_team_secondary_color(stats['team_b'])
    
    # Calcular marges
    scores_a = stats['score_distribution_a']
    scores_b = stats['score_distribution_b']
    margins = [a - b for a, b in zip(scores_a, scores_b)]
    
    # Categoritzar marges
    categories = {
        f"{stats['team_b'][:12]}\n+10 o més": sum(1 for m in margins if m <= -10),
        f"{stats['team_b'][:12]}\n+5 a +9": sum(1 for m in margins if -9 <= m <= -5),
        f"{stats['team_b'][:12]}\n+1 a +4": sum(1 for m in margins if -4 <= m <= -1),
        f"{stats['team_a'][:12]}\n+1 a +4": sum(1 for m in margins if 1 <= m <= 4),
        f"{stats['team_a'][:12]}\n+5 a +9": sum(1 for m in margins if 5 <= m <= 9),
        f"{stats['team_a'][:12]}\n+10 o més": sum(1 for m in margins if m >= 10),
    }
    
    total = sum(categories.values())
    percentages = {k: (v / total * 100) if total > 0 else 0 for k, v in categories.items()}
    
    # Colors: 3 tons de l'equip B + 3 tons de l'equip A
    def lighten_color(hex_color, factor=0.3):
        """Fa el color més clar"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return f'#{r:02x}{g:02x}{b:02x}'
    
    colors = [
        lighten_color(color_b, 0.4),  # Més clar
        lighten_color(color_b, 0.2),
        color_b,                       # Original
        color_a,                       # Original
        lighten_color(color_a, 0.2),
        lighten_color(color_a, 0.4),  # Més clar
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(percentages.keys()),
            y=list(percentages.values()),
            marker_color=colors,
            text=[f"{p:.1f}%" for p in percentages.values()],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>%{y:.1f}% dels partits<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Probabilitat per Marge de Victòria",
        yaxis_title="% de simulacions",
        height=400,
        font=dict(family="Inter, sans-serif"),
        showlegend=False,
        yaxis=dict(range=[0, max(percentages.values()) * 1.15 if percentages else 100])
    )
    
    return fig


def plot_possessions(stats):
    """Gràfic de possessions amb colors dels equips."""
    import numpy as np
    
    color_a = get_team_primary_color(stats['team_a'])
    color_b = get_team_primary_color(stats['team_b'])
    
    if color_a.lower() == color_b.lower():
        color_b = get_team_secondary_color(stats['team_b'])
    
    fig = go.Figure()
    
    poss_a = stats['possessions_distribution_a']
    poss_b = stats['possessions_distribution_b']
    
    min_val = min(min(poss_a), min(poss_b))
    max_val = max(max(poss_a), max(poss_b))
    
    fig.add_trace(go.Histogram(
        x=poss_a,
        name=stats['team_a'][:20],
        marker_color=color_a,
        opacity=0.7,
        xbins=dict(start=min_val-0.5, end=max_val+0.5, size=1),
        hovertemplate='<b>%{x}</b> possessions<br>Freqüència: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Histogram(
        x=poss_b,
        name=stats['team_b'][:20],
        marker_color=color_b,
        opacity=0.7,
        xbins=dict(start=min_val-0.5, end=max_val+0.5, size=1),
        hovertemplate='<b>%{x}</b> possessions<br>Freqüència: %{y}<extra></extra>'
    ))
    
    mean_a = np.mean(poss_a)
    mean_b = np.mean(poss_b)
    
    fig.add_vline(
        x=mean_a, 
        line_dash="dash", 
        line_color=color_a,
        line_width=2,
        annotation_text=f"Mitjana {stats['team_a'][:15]}: {mean_a:.1f}",
        annotation_position="top left"
    )
    fig.add_vline(
        x=mean_b, 
        line_dash="dash", 
        line_color=color_b,
        line_width=2,
        annotation_text=f"Mitjana {stats['team_b'][:15]}: {mean_b:.1f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="Distribució de Possessions",
        xaxis_title="Possessions per partit",
        yaxis_title="Freqüència",
        barmode='overlay',
        height=400,
        font=dict(family="Inter, sans-serif"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def main():
    """Funció principal de l'aplicació."""
    
    # Inicialitzar estat
    init_session_state()
    
    # Header professional amb subtítol
    st.markdown('<h1 class="main-header">🏀 NBA Game Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Sistema de predicció amb cadenes de Markov i simulació Monte Carlo</p>', unsafe_allow_html=True)
    
    # Carregar dades
    players_df, lineups_df, teams = load_data()
    
    if players_df is None:
        st.error("No s'han pogut carregar les dades. Verifica que existeixin els CSV.")
        return
    
    # Sidebar - Disseny modern i professional
    with st.sidebar:
        # Logo NBA centrat
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://upload.wikimedia.org/wikipedia/en/0/03/National_Basketball_Association_logo.svg", width=80)
        
        st.markdown("<h2 style='text-align: center; color: white; margin-top: 0;'>Configuració</h2>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 0.5rem 0; border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
        
        # Secció Simulació
        st.markdown("### 🎲 Paràmetres de Simulació")
        
        # Usar SELECTBOX en lloc de slider per al nombre de simulacions
        n_simulations_options = {
            "1.000 (Ràpid)": 1000,
            "2.500 (Recomanat)": 2500,
            "5.000 (Precís)": 5000,
            "7.500 (Molt Precís)": 7500,
            "10.000 (Màxim)": 10000
        }
        
        n_sim_label = st.selectbox(
            "Simulacions Monte Carlo",
            options=list(n_simulations_options.keys()),
            index=0,
            help="Més simulacions = més precisió però més temps."
        )
        n_simulations = n_simulations_options[n_sim_label]
        
        # Indicador visual del temps estimat amb badge
        time_estimate = n_simulations * 0.01
        if time_estimate < 15:
            time_icon = "⚡"
            time_label = "Ràpid"
            time_bg = "#00C853"
        elif time_estimate < 50:
            time_icon = "⏱️"
            time_label = "Moderat"
            time_bg = "#FF9800"
        else:
            time_icon = "🐌"
            time_label = "Lent"
            time_bg = "#D32F2F"
        
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.1); padding: 0.6rem; border-radius: 8px; text-align: center; margin-top: 0.5rem; border: 1px solid rgba(255,255,255,0.2);'>
            <span style='background: {time_bg}; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600;'>
                {time_icon} {time_label} (~{time_estimate:.0f}s)
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Home Court Advantage - DISSENY NOU AMB SELECTBOX
        st.markdown("### 🏠 Home Court Advantage")
        
        # Selectbox per activar/desactivar (més fiable que toggle)
        home_court_status = st.selectbox(
            "Estat",
            options=["✅ Activat", "❌ Desactivat"],
            index=0,
            label_visibility="collapsed",
            help="Els equips locals guanyen ~60% dels partits NBA. L'avantatge s'aplica dins de la matriu de transició (factor multiplicatiu +2% eficiència + 0.5% pace)"
        )
        
        home_court_advantage = (home_court_status == "✅ Activat")
        
        if home_court_advantage:
            # Number input per als punts
            home_advantage_points = st.number_input(
                "Punts d'avantatge",
                min_value=0.0,
                max_value=6.0,
                value=2.8,
                step=0.5,
                help="Calibratge típic NBA: 2-3 punts. S'aplica internament com a factor multiplicatiu",
                format="%.1f"
            )
            
            # Card informatiu actiu
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.15); padding: 0.7rem; border-radius: 10px; text-align: center; color: white; border: 1px solid rgba(255,255,255,0.3); margin-top: 0.5rem;'>
                <div style='font-size: 0.75rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 1px;'>Avantatge actiu</div>
                <div style='font-size: 1.3rem; font-weight: 700; margin-top: 0.2rem;'>
                    +{home_advantage_points:.1f} punts
                </div>
                <div style='font-size: 0.75rem; opacity: 0.8;'>per l'equip local (dins matriu)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            home_advantage_points = 0.0
            # Card informatiu inactiu
            st.markdown("""
            <div style='background: rgba(255,255,255,0.05); padding: 0.7rem; border-radius: 10px; text-align: center; color: rgba(255,255,255,0.6); border: 1px solid rgba(255,255,255,0.1); margin-top: 0.5rem;'>
                <div style='font-size: 0.85rem;'>❌ Sense avantatge local</div>
                <div style='font-size: 0.75rem; opacity: 0.7; margin-top: 0.3rem;'>Camp neutral</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Opcions avançades
        with st.expander("⚙️ Opcions avançades"):
            show_details = st.checkbox("Mostrar detalls del partit", value=True)
            st.caption("Mostra possessions, jugadors, etc.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Info del projecte
        st.markdown("### 📖 Sobre el Projecte")
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.2);'>
            <p style='margin: 0; font-size: 0.85rem; line-height: 1.5; color: white;'>
                <strong>TFG - Grau en Matemàtica Computacional</strong><br>
                <span style='color: rgba(255,255,255,0.7);'>UAB · 2025</span>
            </p>
            <hr style='margin: 0.5rem 0; border-color: rgba(255,255,255,0.2);'>
            <p style='margin: 0; font-size: 0.8rem; color: rgba(255,255,255,0.7); line-height: 1.6;'>
                Sistema de predicció NBA basat en:
            </p>
            <ul style='margin: 0.3rem 0 0 0; padding-left: 1.2rem; font-size: 0.8rem; color: rgba(255,255,255,0.8);'>
                <li>Cadenes de Markov (17 estats)</li>
                <li>Pace dinàmic per quintet</li>
                <li>Impacte defensiu bidireccional</li>
                <li>Simulació Monte Carlo</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Pestanyes principals
    tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Configuració", "🎲 Simulació", "📊 Resultats", "🔢 Matrius"])
    
    # ========== TAB 1: CONFIGURACIÓ ==========
    with tab1:
        st.header("Configuració d'Equips i Jugadors")
        
        col1, col2 = st.columns(2)
        
        # Equip A
        with col1:
            
            team_a = st.selectbox(
                "Selecciona equip A",
                teams,
                key="team_a_select"
            )
            
            if team_a:
                # Mostrar escut de l'equip amb el seu color
                logo_html = get_team_logo_html(team_a, size=80)
                team_a_color = get_team_primary_color(team_a)
                st.markdown(f"""
                <div style='display: flex; align-items: center; gap: 1rem; padding: 1rem; background: white; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); margin: 0.5rem 0; border-left: 4px solid {team_a_color};'>
                    {logo_html}
                    <div>
                        <div style='font-size: 1.2rem; font-weight: 700; color: #1A1A2E;'>{team_a}</div>
                        <div style='font-size: 0.85rem; color: #6C757D;'>Equip Local</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.team_a_players = render_player_selector(
                    team_a, players_df, 'a'
                )
                
                st.markdown("---")
                display_team_summary(team_a, st.session_state.team_a_players)
        
        # Equip B
        with col2:
            
            team_b = st.selectbox(
                "Selecciona equip B",
                teams,
                key="team_b_select"
            )
            
            if team_b:
                # Mostrar escut de l'equip amb el seu color
                logo_html = get_team_logo_html(team_b, size=80)
                team_b_color = get_team_primary_color(team_b)
                st.markdown(f"""
                <div style='display: flex; align-items: center; gap: 1rem; padding: 1rem; background: white; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); margin: 0.5rem 0; border-left: 4px solid {team_b_color};'>
                    {logo_html}
                    <div>
                        <div style='font-size: 1.2rem; font-weight: 700; color: #1A1A2E;'>{team_b}</div>
                        <div style='font-size: 0.85rem; color: #6C757D;'>Equip Visitant</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.team_b_players = render_player_selector(
                    team_b, players_df, 'b'
                )
                
                st.markdown("---")
                display_team_summary(team_b, st.session_state.team_b_players)
    
    # ========== TAB 2: SIMULACIÓ ==========
    with tab2:
        st.header("Simulació de Partits")
        
        # Verificar configuració
        if not st.session_state.team_a_players or not st.session_state.team_b_players:
            st.warning("⚠️ Primer configura els equips a la pestanya 'Configuració'")
            return
        
        # ============================================
        # VALIDACIÓ DE MINUTS - Han de sumar 240
        # ============================================
        MINUTS_REQUERITS = 240  # 5 jugadors × 48 minuts
        
        total_min_a = sum(st.session_state.team_a_players.values())
        total_min_b = sum(st.session_state.team_b_players.values())
        
        # Mostrar resum de minuts amb estat visual
        col_a, col_b = st.columns(2)
        
        with col_a:
            diff_a = total_min_a - MINUTS_REQUERITS
            if total_min_a == MINUTS_REQUERITS:
                color_a = "#00C853"  # Verd
                icon_a = "✅"
                msg_a = "Correcte"
            elif total_min_a > MINUTS_REQUERITS:
                color_a = "#D32F2F"  # Vermell
                icon_a = "❌"
                msg_a = f"Sobren {diff_a} min"
            else:
                color_a = "#FF9800"  # Taronja
                icon_a = "⚠️"
                msg_a = f"Falten {abs(diff_a)} min"
            
            # Obtenir escut de l'equip A
            logo_a_small = get_team_logo_html(team_a, size=50)
            
            st.markdown(f"""
            <div style='background: white; padding: 1rem 1.5rem; border-radius: 12px; border-left: 4px solid {color_a}; box-shadow: 0 2px 4px rgba(0,0,0,0.08); margin-bottom: 1rem;'>
                <div style='display: flex; justify-content: space-between; align-items: center; gap: 1rem;'>
                    <div>{logo_a_small}</div>
                    <div style='flex: 1;'>
                        <div style='font-size: 0.8rem; color: #6C757D; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;'>
                            {team_a}
                        </div>
                        <div style='font-size: 1.8rem; font-weight: 700; color: {color_a}; line-height: 1.2;'>
                            {total_min_a} / 240 min
                        </div>
                        <div style='font-size: 0.85rem; color: {color_a}; font-weight: 600;'>
                            {icon_a} {msg_a}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            diff_b = total_min_b - MINUTS_REQUERITS
            if total_min_b == MINUTS_REQUERITS:
                color_b = "#00C853"
                icon_b = "✅"
                msg_b = "Correcte"
            elif total_min_b > MINUTS_REQUERITS:
                color_b = "#D32F2F"
                icon_b = "❌"
                msg_b = f"Sobren {diff_b} min"
            else:
                color_b = "#FF9800"
                icon_b = "⚠️"
                msg_b = f"Falten {abs(diff_b)} min"
            
            # Obtenir escut de l'equip B
            logo_b_small = get_team_logo_html(team_b, size=50)
            
            st.markdown(f"""
            <div style='background: white; padding: 1rem 1.5rem; border-radius: 12px; border-left: 4px solid {color_b}; box-shadow: 0 2px 4px rgba(0,0,0,0.08); margin-bottom: 1rem;'>
                <div style='display: flex; justify-content: space-between; align-items: center; gap: 1rem;'>
                    <div>{logo_b_small}</div>
                    <div style='flex: 1;'>
                        <div style='font-size: 0.8rem; color: #6C757D; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;'>
                            {team_b}
                        </div>
                        <div style='font-size: 1.8rem; font-weight: 700; color: {color_b}; line-height: 1.2;'>
                            {total_min_b} / 240 min
                        </div>
                        <div style='font-size: 0.85rem; color: {color_b}; font-weight: 600;'>
                            {icon_b} {msg_b}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Comprovació final: només permetre simular si tots dos equips tenen 240 minuts
        minuts_correctes = (total_min_a == MINUTS_REQUERITS and total_min_b == MINUTS_REQUERITS)
        
        if not minuts_correctes:
            st.markdown(f"""
            <div style='background: #FFF3CD; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #FF9800; margin: 1rem 0;'>
                <div style='display: flex; align-items: center; gap: 1rem;'>
                    <span style='font-size: 2rem;'>⚠️</span>
                    <div>
                        <div style='font-weight: 700; color: #856404; font-size: 1.1rem;'>
                            No es pot simular el partit
                        </div>
                        <div style='color: #856404; font-size: 0.9rem; margin-top: 0.3rem;'>
                            Cada equip ha de sumar exactament <strong>240 minuts</strong> (5 jugadors × 48 minuts).
                            Ajusta els minuts dels jugadors a la pestanya <strong>'Configuració'</strong>.
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            return  # Sortir sense mostrar botons
        
        # Botons de simulació (només si els minuts són correctes)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Simular 1 Partit", use_container_width=True, type="primary"):
                with st.spinner("Simulant partit..."):
                    try:
                        # Carregar equips
                        team_a_rotation = load_team_rotation(team_a, max_lineups=20)
                        team_b_rotation = load_team_rotation(team_b, max_lineups=20)
                        
                        # Simular partit regular (48 minuts)
                        simulator = RealisticGameSimulator(team_a_rotation, team_b_rotation)
                        result = simulator.simulate_game(verbose=False, show_quarters=True, show_key_moments=False)
                        
                        # PRÒRROGUES INFINITES FINS QUE HI HAGI GUANYADOR
                        overtime_count = 0
                        while result['winner'] == 'Tie':
                            overtime_count += 1
                            st.info(f"⏱️ EMPAT {result['score_a']}-{result['score_b']}! Pròrroga {overtime_count} (5 minuts)...")
                            
                            # Simular pròrroga de 5 minuts
                            # Nota: simulate_game simula 48min, però per pròrroga necessitem 5min
                            # Simulem un partit curt i ajustem
                            ot_simulator = RealisticGameSimulator(team_a_rotation, team_b_rotation)
                            
                            # Simular aprox 10-12 possessions (5 min = ~10 possessions)
                            ot_score_a = result['score_a']
                            ot_score_b = result['score_b']
                            
                            for _ in range(12):  # ~12 possessions en 5 minuts
                                # Possessió equip A
                                states_a, points_a, _ = team_a_rotation.markov_lineups[0]['lineup'].simulate_possession()
                                ot_score_a += points_a
                                
                                # Possessió equip B
                                states_b, points_b, _ = team_b_rotation.markov_lineups[0]['lineup'].simulate_possession()
                                ot_score_b += points_b
                            
                            # Actualitzar marcadors
                            result['score_a'] = ot_score_a
                            result['score_b'] = ot_score_b
                            
                            # Determinar guanyador
                            if ot_score_a > ot_score_b:
                                result['winner'] = 'A'
                            elif ot_score_b > ot_score_a:
                                result['winner'] = 'B'
                            else:
                                result['winner'] = 'Tie'  # Continua el bucle
                            
                            # Protecció contra bucle infinit (màxim 10 pròrrogues)
                            if overtime_count >= 10:
                                st.warning("⚠️ S'han superat 10 pròrrogues. Forçant guanyador...")
                                if result['score_a'] >= result['score_b']:
                                    result['winner'] = 'A'
                                    result['score_a'] += 1  # Assegurar que hi ha diferència
                                else:
                                    result['winner'] = 'B'
                                    result['score_b'] += 1
                                break
                        
                        st.session_state.simulation_result = result
                        
                        if overtime_count > 0:
                            st.success(f"✅ Partit finalitzat després de {overtime_count} {'pròrroga' if overtime_count == 1 else 'pròrrogues'}!")
                        else:
                            st.success("✅ Simulació completada!")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        with col2:
            if st.button("🎲 Simular Monte Carlo", use_container_width=True, type="primary"):
                with st.spinner(f"Executant {n_simulations} simulacions..."):
                    try:
                        # Carregar equips
                        team_a_rotation = load_team_rotation(team_a, max_lineups=20)
                        team_b_rotation = load_team_rotation(team_b, max_lineups=20)
                        
                        # NOU: Configurar Monte Carlo amb HCA integrat dins la matriu
                        # L'avantatge ja no se suma a posteriori, sinó que afecta directament
                        # les probabilitats d'encert del quintet local via home_court_factor
                        # i el seu pace lleugerament incrementat.
                        if home_court_advantage:
                            predictor = MonteCarloGamePredictor(
                                team_a_rotation, 
                                team_b_rotation,
                                home_team='a',  # team_a és el local
                                home_court_pts=home_advantage_points  # Calibratge HCA
                            )
                        else:
                            # Sense avantatge: equips neutres (cap dels dos és local)
                            predictor = MonteCarloGamePredictor(
                                team_a_rotation, 
                                team_b_rotation,
                                home_team=None,
                                home_court_pts=0.0
                            )
                        
                        stats = predictor.run_monte_carlo(
                            n_simulations=n_simulations, 
                            verbose=False
                            # home_advantage ja no es passa aquí (és OBSOLET)
                            # L'avantatge està integrat dins la matriu de transició
                        )
                        
                        st.session_state.mc_stats = stats
                        
                        # Mostrar missatge amb detalls
                        msg = f"✅ {n_simulations} simulacions completades!"
                        if home_court_advantage:
                            msg += f" (Home Court: +{home_advantage_points:.1f} pts per {team_a})"
                        st.success(msg)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        # Mostrar resultats simulació individual
        if st.session_state.simulation_result:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center;'>📊 Resultat Final del Partit</h2>", unsafe_allow_html=True)
            
            result = st.session_state.simulation_result
            
            # Determinar guanyador i colors
            winner_a = result['winner'] == 'A'
            winner_b = result['winner'] == 'B'
            margin = abs(result['score_a'] - result['score_b'])
            
            # Score display amb columnes de Streamlit (més fiable amb logos grans)
            logo_a_html = get_team_logo_html(result['team_a_name'], size=70)
            logo_b_html = get_team_logo_html(result['team_b_name'], size=70)
            
            team_a_color = get_team_primary_color(result['team_a_name'])
            team_b_color = get_team_primary_color(result['team_b_name'])
            
            # Usar columnes per evitar problemes amb HTML llarg
            score_col_a, score_col_div, score_col_b = st.columns([2, 0.3, 2])
            
            with score_col_a:
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem;'>
                    <div style='margin-bottom: 0.5rem; display: flex; justify-content: center;'>{logo_a_html}</div>
                    <div style='font-size: 1.3rem; font-weight: 600; color: {team_a_color if winner_a else "#6C757D"}; margin-bottom: 0.5rem;'>
                        {result['team_a_name']}
                    </div>
                    <div style='font-size: 4rem; font-weight: 800; color: {team_a_color if winner_a else "#9CA3AF"}; line-height: 1;'>
                        {result['score_a']}
                    </div>
                    {f'<div style="margin-top: 0.5rem;"><span style="background: linear-gradient(135deg, #FFD700, #FFA500); color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 700; font-size: 0.85rem;">🏆 GUANYADOR</span></div>' if winner_a else ''}
                </div>
                """, unsafe_allow_html=True)
            
            with score_col_div:
                st.markdown(f"""
                <div style='text-align: center; padding: 2rem 0; font-size: 2.5rem; color: #DEE2E6; font-weight: 300;'>
                    −
                </div>
                """, unsafe_allow_html=True)
            
            with score_col_b:
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem;'>
                    <div style='margin-bottom: 0.5rem; display: flex; justify-content: center;'>{logo_b_html}</div>
                    <div style='font-size: 1.3rem; font-weight: 600; color: {team_b_color if winner_b else "#6C757D"}; margin-bottom: 0.5rem;'>
                        {result['team_b_name']}
                    </div>
                    <div style='font-size: 4rem; font-weight: 800; color: {team_b_color if winner_b else "#9CA3AF"}; line-height: 1;'>
                        {result['score_b']}
                    </div>
                    {f'<div style="margin-top: 0.5rem;"><span style="background: linear-gradient(135deg, #FFD700, #FFA500); color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 700; font-size: 0.85rem;">🏆 GUANYADOR</span></div>' if winner_b else ''}
                </div>
                """, unsafe_allow_html=True)
            
            # Diferència i info ràpida
            st.markdown(f"""
            <div style='text-align: center; margin-top: -1rem; margin-bottom: 1.5rem;'>
                <span style='background: #f8f9fa; padding: 0.5rem 1.5rem; border-radius: 20px; color: #6C757D; font-weight: 600;'>
                    Diferència: <strong style='color: #17408B;'>{margin} punts</strong>
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Stats del partit amb icones
            st.markdown("<h3>📈 Estadístiques del Partit</h3>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-card-title">⚡ Possessions A</div>
                    <div class="info-card-value">{result['possessions_a']}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-card-title">⚡ Possessions B</div>
                    <div class="info-card-value">{result['possessions_b']}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-card-title">🎯 ORtg {result['team_a_name'][:10]}</div>
                    <div class="info-card-value">{result['ortg_a']:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-card-title">🎯 ORtg {result['team_b_name'][:10]}</div>
                    <div class="info-card-value">{result['ortg_b']:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Box score per quarters amb estil
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h3>📊 Marcador per Quarts</h3>", unsafe_allow_html=True)
            
            quarters_df = pd.DataFrame({
                'Quarter': ['Q1', 'Q2', 'Q3', 'Q4', '🏆 TOTAL'],
                result['team_a_name']: [
                    result['quarter_scores'][1]['a'],
                    result['quarter_scores'][2]['a'],
                    result['quarter_scores'][3]['a'],
                    result['quarter_scores'][4]['a'],
                    result['score_a']
                ],
                result['team_b_name']: [
                    result['quarter_scores'][1]['b'],
                    result['quarter_scores'][2]['b'],
                    result['quarter_scores'][3]['b'],
                    result['quarter_scores'][4]['b'],
                    result['score_b']
                ]
            })
            
            # Mostrar dataframe normal (sense estilitzar per evitar dependència de jinja2)
            st.dataframe(quarters_df, use_container_width=True, hide_index=True)
    
    # ========== TAB 3: RESULTATS ==========
    with tab3:
        st.header("Resultats Monte Carlo")
        
        if st.session_state.mc_stats is None:
            st.markdown("""
            <div style='background: #17408B; padding: 3rem 2rem; border-radius: 16px; text-align: center; color: white; margin: 2rem 0;'>
                <h2 style='color: white; margin-bottom: 1rem;'>🎲 Cap simulació executada</h2>
                <p style='font-size: 1.1rem; opacity: 0.9;'>Ves a la pestanya <strong>'Simulació'</strong> i executa una simulació Monte Carlo per veure els resultats aquí.</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        stats = st.session_state.mc_stats
        
        # Banner home court advantage - MONOCROMÀTIC
        # NOU: el HCA està ara integrat dins la matriu de transició
        if stats.get('home_advantage', 0) > 0 or stats.get('home_court_pts', 0) > 0:
            hca_value = stats.get('home_court_pts', stats.get('home_advantage', 0))
            st.markdown(f"""
            <div style='background: #17408B; padding: 1rem 1.5rem; border-radius: 12px; color: white; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
                <div style='display: flex; align-items: center; gap: 1rem;'>
                    <span style='font-size: 2rem;'>🏠</span>
                    <div>
                        <div style='font-weight: 700; font-size: 1.1rem;'>Home Court Advantage Activat</div>
                        <div style='opacity: 0.9; font-size: 0.9rem;'>~+{hca_value:.1f} punts efectius per {stats['team_a']} (integrat dins la matriu Markov)</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # PREDICCIÓ PRINCIPAL - MONOCROMÀTIC
        prob_a = stats['prob_win_a'] * 100
        prob_b = stats['prob_win_b'] * 100
        predicted_winner = stats['team_a'] if prob_a > prob_b else stats['team_b']
        winner_prob = max(prob_a, prob_b)
        
        # Determinar confiança
        if winner_prob >= 70:
            confidence = "Alta"
            conf_color = "#00C853"
            conf_icon = "🟢"
        elif winner_prob >= 55:
            confidence = "Mitjana"
            conf_color = "#FF9800"
            conf_icon = "🟡"
        else:
            confidence = "Baixa"
            conf_color = "#D32F2F"
            conf_icon = "🔴"
        
        # Obtenir escut del guanyador predit
        winner_logo_html = get_team_logo_html(predicted_winner, size=80)
        
        st.markdown(f"""
        <div style='background: white; padding: 2rem; border-radius: 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.1); margin-bottom: 1.5rem; text-align: center;'>
            <div style='color: #6C757D; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; font-weight: 600;'>
                Predicció Monte Carlo · {stats['n_simulations']} simulacions
            </div>
            <div style='margin: 1rem 0;'>
                {winner_logo_html}
            </div>
            <div style='font-size: 2.5rem; font-weight: 800; color: #17408B; margin: 0.5rem 0;'>
                🏆 {predicted_winner}
            </div>
            <div style='font-size: 1.2rem; color: #6C757D; margin-bottom: 1rem;'>
                Probabilitat de guanyar: <strong style='color: #17408B; font-size: 1.5rem;'>{winner_prob:.1f}%</strong>
            </div>
            <span style='background: {conf_color}20; color: {conf_color}; padding: 0.4rem 1rem; border-radius: 20px; font-weight: 600; font-size: 0.9rem;'>
                {conf_icon} Confiança: {confidence}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Barra de probabilitat visual amb COLORS dels equips
        team_a_color = get_team_primary_color(stats['team_a'])
        team_b_color = get_team_primary_color(stats['team_b'])
        
        # Si són iguals, usar secundari del B
        if team_a_color.lower() == team_b_color.lower():
            team_b_color = get_team_secondary_color(stats['team_b'])
        
        st.markdown(f"""
        <div class="win-prob-container">
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                <div>
                    <span style='font-weight: 700; color: {team_a_color}; font-size: 1.1rem;'>{stats['team_a']}</span>
                    <span style='color: #6C757D; margin-left: 0.5rem;'>{prob_a:.1f}%</span>
                </div>
                <div style='text-align: right;'>
                    <span style='color: #6C757D; margin-right: 0.5rem;'>{prob_b:.1f}%</span>
                    <span style='font-weight: 700; color: {team_b_color}; font-size: 1.1rem;'>{stats['team_b']}</span>
                </div>
            </div>
            <div class="win-prob-bar">
                <div style="width: {prob_a}%; background: {team_a_color}; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 0.95rem;">
                    {prob_a:.0f}%
                </div>
                <div style="width: {prob_b}%; background: {team_b_color}; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 0.95rem;">
                    {prob_b:.0f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Mètriques principals - TOTES BLAUES
        st.markdown("<h3>📊 Mètriques Clau</h3>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-card-title">📈 Score Esperat A</div>
                <div class="info-card-value" style='color: #17408B;'>{stats['expected_score_a']:.1f}</div>
                <div style='font-size: 0.75rem; color: #6C757D; margin-top: 0.3rem;'>±{stats['std_score_a']:.1f} pts</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-card-title">📈 Score Esperat B</div>
                <div class="info-card-value" style='color: #17408B;'>{stats['expected_score_b']:.1f}</div>
                <div style='font-size: 0.75rem; color: #6C757D; margin-top: 0.3rem;'>±{stats['std_score_b']:.1f} pts</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-card-title">🎯 Total Esperat</div>
                <div class="info-card-value" style='color: #17408B;'>{stats['expected_total']:.1f}</div>
                <div style='font-size: 0.75rem; color: #6C757D; margin-top: 0.3rem;'>punts totals</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            over_color = "#00C853" if stats['prob_over_220'] > 0.5 else "#D32F2F"
            st.markdown(f"""
            <div class="info-card" style='border-left-color: {over_color};'>
                <div class="info-card-title">🎲 Over 220</div>
                <div class="info-card-value" style='color: {over_color};'>{stats['prob_over_220']*100:.1f}%</div>
                <div style='font-size: 0.75rem; color: #6C757D; margin-top: 0.3rem;'>probabilitat</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # GRÀFICS PRINCIPALS - FILA 1
        st.markdown("<h3>📊 Gràfics Principals</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_win_probability(stats), use_container_width=True)
            st.plotly_chart(plot_score_distributions(stats), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_total_points(stats), use_container_width=True)
            st.plotly_chart(plot_possessions(stats), use_container_width=True)
        
        # GRÀFICS AVANÇATS - FILA 2 (NOVES VISUALITZACIONS)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3>📈 Anàlisi Avançada</h3>", unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.plotly_chart(plot_margin_distribution(stats), use_container_width=True)
        
        with col4:
            st.plotly_chart(plot_offensive_rating(stats), use_container_width=True)
        
        # Gràfic d'escenaris de victòria (full width)
        st.plotly_chart(plot_win_scenarios(stats), use_container_width=True)
        
        # Taula resum
        st.markdown("---")
        st.subheader("📋 Resum Estadístic")
        
        summary_df = pd.DataFrame({
            'Mètrica': [
                'Marcador Esperat',
                'Desviació Estàndard',
                'Interval 95%',
                'Possessions Mitjanes',
                'Pace Mitjà'
            ],
            stats['team_a'][:20]: [
                f"{stats['expected_score_a']:.1f}",
                f"±{stats['std_score_a']:.1f}",
                f"[{stats['ci_95_score_a'][0]:.0f}, {stats['ci_95_score_a'][1]:.0f}]",
                f"{stats['expected_possessions_a']:.1f}",
                f"{stats['expected_pace']:.1f}"
            ],
            stats['team_b'][:20]: [
                f"{stats['expected_score_b']:.1f}",
                f"±{stats['std_score_b']:.1f}",
                f"[{stats['ci_95_score_b'][0]:.0f}, {stats['ci_95_score_b'][1]:.0f}]",
                f"{stats['expected_possessions_b']:.1f}",
                f"{stats['expected_pace']:.1f}"
            ]
        })
        
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Botó descarregar
        st.markdown("---")
        if st.button("💾 Descarregar Informe JSON", use_container_width=True):
            json_data = {
                'n_simulations': stats['n_simulations'],
                'team_a': stats['team_a'],
                'team_b': stats['team_b'],
                'probabilities': {
                    'win_a': float(stats['prob_win_a']),
                    'win_b': float(stats['prob_win_b'])
                },
                'expected_scores': {
                    'team_a': float(stats['expected_score_a']),
                    'team_b': float(stats['expected_score_b'])
                },
                'possessions': {
                    'team_a': float(stats['expected_possessions_a']),
                    'team_b': float(stats['expected_possessions_b']),
                    'pace': float(stats['expected_pace'])
                }
            }
            
            st.download_button(
                label="📥 Descarregar JSON",
                data=json.dumps(json_data, indent=2),
                file_name="prediction_results.json",
                mime="application/json"
            )
    
    # ========== TAB 4: MATRIUS DE TRANSICIÓ ==========
    with tab4:
        st.header("🔢 Matrius de Transició de Markov")
        
        if not st.session_state.team_a_players or not st.session_state.team_b_players:
            st.info("👆 Configura els equips a la pestanya 'Configuració' per veure les matrius")
        else:
            try:
                # Carregar dades de jugadors
                players_df, _, _ = load_data()
                
                # Selector d'equip per veure
                team_to_show = st.selectbox(
                    "Selecciona equip per veure matrius:",
                    [team_a, team_b],
                    key="matrix_team_selector"
                )
                
                # Obtenir jugadors de l'equip seleccionat
                if team_to_show == team_a:
                    selected_players = st.session_state.team_a_players
                else:
                    selected_players = st.session_state.team_b_players
                
                if not selected_players:
                    st.warning("No hi ha jugadors seleccionats per aquest equip")
                else:
                    # Capçalera amb ESCUT de l'equip (en lloc de pilota)
                    team_logo_html = get_team_logo_html(team_to_show, size=50)
                    team_color = get_team_primary_color(team_to_show)
                    
                    st.markdown(f"""
                    <div style='display: flex; align-items: center; gap: 1rem; padding: 1rem; background: white; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); margin: 1rem 0; border-left: 4px solid {team_color};'>
                        <div>{team_logo_html}</div>
                        <div>
                            <div style='font-size: 1.5rem; font-weight: 700; color: #1A1A2E;'>{team_to_show}</div>
                            <div style='font-size: 0.9rem; color: #6C757D;'>{len(selected_players)} jugadors seleccionats</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Selector de jugador
                    player_names = list(selected_players.keys())
                    selected_player = st.selectbox(
                        "Selecciona jugador per veure la seva matriu:",
                        ["📊 Matriu Combinada del Quintet"] + player_names,
                        key="player_selector"
                    )
                    
                    # Estats del model (ORDRE CORRECTE segons GameStates)
                    states = [
                        'START',           # 0
                        'Corner 3',        # 1
                        'Above Break 3',   # 2
                        'Mid-Range',       # 3
                        'Long Mid',        # 4
                        'Paint',           # 5
                        'Layup',           # 6
                        'Dunk',            # 7
                        'TOV',             # 8
                        'Foul',            # 9
                        'FT',              # 10
                        'Steal',           # 11
                        'Block',           # 12
                        'Deflection',      # 13
                        'OREB',            # 14
                        'DREB',            # 15
                        'END'              # 16
                    ]
                    
                    if selected_player == "📊 Matriu Combinada del Quintet":
                        # Mostrar matriu combinada
                        st.markdown("### 📊 Matriu Combinada del Quintet")
                        st.info("Aquesta és la matriu de transició mitjana dels jugadors seleccionats, ponderada pels seus minuts de joc")
                        
                        # Calcular matriu combinada
                        total_minutes = sum(selected_players.values())
                        combined_matrix = None
                        
                        for player_name, minutes in selected_players.items():
                            # Buscar jugador al CSV
                            player_data = players_df[players_df['player_name'] == player_name]
                            
                            if len(player_data) == 0:
                                st.warning(f"⚠️ No s'han trobat dades per {player_name}")
                                continue
                            
                            player_row = player_data.iloc[0]
                            
                            # Obtenir matriu del jugador (des del model)
                            try:
                                # Importar AdvancedPlayerFromData
                                import sys
                                import os
                                
                                # Afegir directori actual al path si no hi és
                                current_dir = os.path.dirname(os.path.abspath(__file__))
                                if current_dir not in sys.path:
                                    sys.path.insert(0, current_dir)
                                
                                from complete_markov_model import AdvancedPlayerFromData
                                
                                # Convertir row a diccionari
                                player_dict = player_row.to_dict()
                                player_obj = AdvancedPlayerFromData(player_dict)
                                player_matrix = player_obj._build_transition_matrix()
                                
                                # Ponderar per minuts
                                weight = minutes / total_minutes
                                
                                if combined_matrix is None:
                                    combined_matrix = player_matrix * weight
                                else:
                                    combined_matrix += player_matrix * weight
                                    
                            except ImportError as ie:
                                st.error(f"⚠️ Error d'importació per {player_name}: {ie}")
                                st.info("""
                                💡 **Solució**: Assegura't que l'arxiu `complete_markov_model.py` 
                                està al mateix directori que `tfg_web_app.py`
                                """)
                                continue
                            except Exception as e:
                                st.warning(f"⚠️ No s'ha pogut carregar matriu de {player_name}: {e}")
                                continue
                        
                        if combined_matrix is not None:
                            # Crear DataFrame
                            if combined_matrix.shape[0] == 17:
                                df_matrix = pd.DataFrame(
                                    combined_matrix,
                                    columns=states,
                                    index=states
                                )
                            else:
                                df_matrix = pd.DataFrame(combined_matrix)
                            
                            # Mostrar amb heatmap Plotly
                            st.markdown("**Matriu de Transició Combinada 17×17**")
                            
                            st.markdown("""
                            **📋 Matriu del Quintet (Mitjana Ponderada):**
                            - Combina les matrius dels 5 jugadors ponderant pels seus minuts
                            - Representa el comportament col·lectiu de l'equip
                            """)
                            
                            # Crear heatmap amb seccions
                            fig = go.Figure()
                            
                            fig.add_trace(go.Heatmap(
                                z=combined_matrix,
                                x=states,
                                y=states,
                                colorscale='RdYlGn',
                                zmin=0,
                                zmax=0.5,
                                text=[[f"{val:.2f}" for val in row] for row in combined_matrix],
                                texttemplate='%{text}',
                                textfont={"size": 9},
                                colorbar=dict(title="Probabilitat"),
                                hovertemplate='<b>De:</b> %{y}<br><b>A:</b> %{x}<br><b>Prob:</b> %{z:.3f}<extra></extra>'
                            ))
                            
                            # Afegir línies divisòries
                            shapes = [
                                dict(type="line", x0=-0.5, x1=16.5, y0=0.5, y1=0.5, 
                                     line=dict(color="black", width=2)),
                                dict(type="line", x0=-0.5, x1=16.5, y0=7.5, y1=7.5, 
                                     line=dict(color="black", width=2)),
                                dict(type="line", x0=0.5, x1=0.5, y0=-0.5, y1=16.5, 
                                     line=dict(color="black", width=2)),
                                dict(type="line", x0=7.5, x1=7.5, y0=-0.5, y1=16.5, 
                                     line=dict(color="black", width=2)),
                            ]
                            
                            fig.update_layout(
                                title={
                                    'text': f"Cadena de Markov - {team_to_show} (Quintet)<br><sub>Matriu Combinada Ponderada per Minuts</sub>",
                                    'x': 0.5,
                                    'xanchor': 'center'
                                },
                                xaxis=dict(
                                    title="<b>Estat Següent (On va)</b>",
                                    side='bottom',
                                    tickangle=-45
                                ),
                                yaxis=dict(
                                    title="<b>Estat Actual (On està)</b>",
                                    autorange='reversed'
                                ),
                                height=800,
                                width=1000,
                                shapes=shapes
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # També mostrar com a taula amb seccions
                            with st.expander("📋 Veure Matriu Combinada Detallada"):
                                st.markdown("### 🎯 DISTRIBUCIÓ INICIAL DEL QUINTET")
                                start_combined = pd.DataFrame({
                                    'Acció': ['Corner 3', 'Above Break 3', 'Mid-Range', 'Long Mid', 
                                             'Paint', 'Layup', 'Dunk', 'TOV'],
                                    'Probabilitat': [f"{combined_matrix[0, i]:.3f}" for i in [1,2,3,4,5,6,7,8]],
                                    'Percentatge': [f"{combined_matrix[0, i]*100:.1f}%" for i in [1,2,3,4,5,6,7,8]]
                                })
                                st.dataframe(start_combined, use_container_width=True, hide_index=True)
                                
                                st.markdown("---")
                                st.markdown("### 🏀 EFECTIVITAT PER ZONA")
                                effectiveness = pd.DataFrame({
                                    'Zona': states[1:8],
                                    '% Conversió': [f"{combined_matrix[i, 16]*100:.1f}%" for i in range(1, 8)],
                                    '% Rebot Defensiu': [f"{combined_matrix[i, 15]*100:.1f}%" for i in range(1, 8)],
                                    '% Rebot Ofensiu': [f"{combined_matrix[i, 14]*100:.1f}%" for i in range(1, 8)]
                                })
                                st.dataframe(effectiveness, use_container_width=True, hide_index=True)
                                
                                st.markdown("---")
                                st.markdown("### 📊 Matriu Completa")
                                st.dataframe(df_matrix, height=600, use_container_width=True)
                            
                            # Stats del quintet
                            st.markdown("---")
                            st.subheader("📊 Estadístiques del Quintet")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Minuts", f"{total_minutes}")
                            with col2:
                                st.metric("Jugadors", f"{len(selected_players)}")
                            with col3:
                                avg_minutes = total_minutes / len(selected_players)
                                st.metric("Mitjana Minuts/Jugador", f"{avg_minutes:.1f}")
                            with col4:
                                st.metric("Dimensions Matriu", f"{combined_matrix.shape[0]}×{combined_matrix.shape[1]}")
                            
                            # Explicació
                            st.markdown("---")
                            st.info("""
                            📖 **Com llegir la matriu combinada:**
                            - Aquesta matriu representa el comportament mitjà del quintet
                            - Ponderada pels minuts de joc de cada jugador
                            - Cada fila = estat actual, Cada columna = estat següent
                            - Els valors són probabilitats de transició (0 a 1)
                            - 🟢 Verd = alta probabilitat, 🟡 Groc = mitjana, 🔴 Vermell = baixa probabilitat
                            
                            **Exemple de lectura:**
                            - Fila "Corner 3" → Columna "Made 3": Probabilitat de convertir un tir de 3 punts des del racó
                            - Fila "Layup" → Columna "Made 2": Probabilitat de convertir una entrada a cistella
                            - Qualsevol estat → Columna "TOV": Probabilitat de perdre la pilota
                            """)
                        else:
                            st.error("❌ No s'ha pogut calcular la matriu combinada")
                    
                    else:
                        # Mostrar matriu individual del jugador
                        # Capçalera amb FOTO del jugador
                        player_photo_html = get_player_photo_html(selected_player, size=60, round_img=True)
                        team_color = get_team_primary_color(team_to_show)
                        
                        st.markdown(f"""
                        <div style='display: flex; align-items: center; gap: 1rem; padding: 1rem; background: white; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); margin: 1rem 0; border-left: 4px solid {team_color};'>
                            <div>{player_photo_html}</div>
                            <div>
                                <div style='font-size: 1.5rem; font-weight: 700; color: #1A1A2E;'>{selected_player}</div>
                                <div style='font-size: 0.9rem; color: #6C757D;'>{team_to_show}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Buscar jugador al CSV
                        player_data = players_df[players_df['player_name'] == selected_player]
                        
                        if len(player_data) == 0:
                            st.error(f"❌ No s'han trobat dades per {selected_player}")
                        else:
                            player_row = player_data.iloc[0]
                            
                            # Mostrar stats del jugador
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("MPG", f"{player_row.get('MPG', 0):.1f}")
                            with col2:
                                # PPG = Points Per Game (calculat des de PTS totals / GP)
                                pts_total = player_row.get('PTS', 0)
                                games = player_row.get('GP', 1)
                                ppg = pts_total / games if games > 0 else 0
                                st.metric("PPG", f"{ppg:.1f}")
                            with col3:
                                st.metric("Minuts assignats", f"{selected_players[selected_player]}")
                            with col4:
                                weight = selected_players[selected_player] / sum(selected_players.values())
                                st.metric("Pes al quintet", f"{weight:.1%}")
                            
                            st.markdown("---")
                            st.subheader("🔢 Matriu de Transició de Markov Individual")
                            
                            try:
                                # Importar i crear objecte jugador
                                import sys
                                import os
                                current_dir = os.path.dirname(os.path.abspath(__file__))
                                if current_dir not in sys.path:
                                    sys.path.insert(0, current_dir)
                                
                                from complete_markov_model import AdvancedPlayerFromData
                                
                                # Convertir row a diccionari i crear jugador
                                player_dict = player_row.to_dict()
                                player_obj = AdvancedPlayerFromData(player_dict)
                                
                                # Generar matriu de transició
                                player_matrix = player_obj._build_transition_matrix()
                                
                                # Crear DataFrame amb noms d'estats
                                if player_matrix.shape[0] == 17:
                                    df_matrix = pd.DataFrame(
                                        player_matrix,
                                        columns=states,
                                        index=states
                                    )
                                else:
                                    df_matrix = pd.DataFrame(player_matrix)
                                
                                # Mostrar amb heatmap
                                st.markdown("**Matriu de Transició 17×17**")
                                
                                # Crear visualització millorada amb seccions
                                st.markdown("""
                                **📋 Estructura de la Matriu:**
                                - **Secció INICI**: Fila START → On va quan comença la possessió
                                - **Secció ZONES DE TIR**: Files 1-7 → Què passa després de cada tipus de tir
                                - **Secció ESDEVENIMENTS**: Files 8-16 → Pèrdues, rebots, faltes, etc.
                                """)
                                
                                # Crear heatmap amb anotacions millorades
                                fig = go.Figure()
                                
                                # Afegir heatmap principal
                                fig.add_trace(go.Heatmap(
                                    z=player_matrix,
                                    x=states,
                                    y=states,
                                    colorscale='RdYlGn',
                                    zmin=0,
                                    zmax=0.5,
                                    text=[[f"{val:.2f}" for val in row] for row in player_matrix],
                                    texttemplate='%{text}',
                                    textfont={"size": 9},
                                    colorbar=dict(title="Probabilitat"),
                                    hovertemplate='<b>De:</b> %{y}<br><b>A:</b> %{x}<br><b>Prob:</b> %{z:.3f}<extra></extra>'
                                ))
                                
                                # Afegir línies divisòries per seccions
                                shapes = [
                                    # Separar START de zones de tir
                                    dict(type="line", x0=-0.5, x1=16.5, y0=0.5, y1=0.5, 
                                         line=dict(color="black", width=2)),
                                    # Separar zones de tir d'esdeveniments
                                    dict(type="line", x0=-0.5, x1=16.5, y0=7.5, y1=7.5, 
                                         line=dict(color="black", width=2)),
                                    # Separar columnes
                                    dict(type="line", x0=0.5, x1=0.5, y0=-0.5, y1=16.5, 
                                         line=dict(color="black", width=2)),
                                    dict(type="line", x0=7.5, x1=7.5, y0=-0.5, y1=16.5, 
                                         line=dict(color="black", width=2)),
                                ]
                                
                                fig.update_layout(
                                    title={
                                        'text': f"Cadena de Markov - {selected_player}<br><sub>Matriu de Probabilitats de Transició 17×17</sub>",
                                        'x': 0.5,
                                        'xanchor': 'center'
                                    },
                                    xaxis=dict(
                                        title="<b>Estat Següent (On va)</b>",
                                        side='bottom',
                                        tickangle=-45
                                    ),
                                    yaxis=dict(
                                        title="<b>Estat Actual (On està)</b>",
                                        autorange='reversed'
                                    ),
                                    height=800,
                                    width=1000,
                                    shapes=shapes
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Taula amb seccions separades
                                with st.expander("📋 Veure Matriu com a Taula Detallada"):
                                    st.markdown("### 🎯 SECCIÓ 1: INICI DE POSSESSIÓ (Fila START)")
                                    st.markdown("*Mostra la distribució inicial - quines accions prefereix el jugador*")
                                    
                                    start_row = pd.DataFrame({
                                        'Estat Actual': ['START'],
                                        'Corner 3': [f"{player_matrix[0, 1]:.3f}"],
                                        'Above Break 3': [f"{player_matrix[0, 2]:.3f}"],
                                        'Mid-Range': [f"{player_matrix[0, 3]:.3f}"],
                                        'Long Mid': [f"{player_matrix[0, 4]:.3f}"],
                                        'Paint': [f"{player_matrix[0, 5]:.3f}"],
                                        'Layup': [f"{player_matrix[0, 6]:.3f}"],
                                        'Dunk': [f"{player_matrix[0, 7]:.3f}"],
                                        'TOV': [f"{player_matrix[0, 8]:.3f}"],
                                        'Foul': [f"{player_matrix[0, 9]:.3f}"],
                                        'Steal': [f"{player_matrix[0, 11]:.3f}"],
                                        'END': [f"{player_matrix[0, 16]:.3f}"]
                                    })
                                    st.dataframe(start_row, use_container_width=True, hide_index=True)
                                    
                                    st.markdown("---")
                                    st.markdown("### 🏀 SECCIÓ 2: ZONES DE TIR (Files 1-7)")
                                    st.markdown("*Mostra què passa després de tirar des de cada zona*")
                                    
                                    shooting_zones = pd.DataFrame({
                                        'Zona de Tir': states[1:8],
                                        'Converteix (→END)': [f"{player_matrix[i, 16]:.3f}" for i in range(1, 8)],
                                        'Falla (→DREB)': [f"{player_matrix[i, 15]:.3f}" for i in range(1, 8)],
                                        'Rebot Propi (→OREB)': [f"{player_matrix[i, 14]:.3f}" for i in range(1, 8)],
                                        'Falta (→Foul)': [f"{player_matrix[i, 9]:.3f}" for i in range(1, 8)],
                                        'Tap (→Block)': [f"{player_matrix[i, 12]:.3f}" for i in range(1, 8)]
                                    })
                                    st.dataframe(shooting_zones, use_container_width=True, hide_index=True)
                                    
                                    st.markdown("---")
                                    st.markdown("### ⚡ SECCIÓ 3: ESDEVENIMENTS (Files 8-16)")
                                    st.markdown("*Mostra les transicions des d'esdeveniments especials*")
                                    
                                    events_data = []
                                    event_states = [(8, 'TOV'), (9, 'Foul'), (10, 'FT'), (11, 'Steal'), 
                                                   (12, 'Block'), (13, 'Deflection'), (14, 'OREB'), (15, 'DREB')]
                                    
                                    for idx, name in event_states:
                                        # Trobar la transició més probable
                                        max_idx = np.argmax(player_matrix[idx])
                                        max_prob = player_matrix[idx, max_idx]
                                        events_data.append({
                                            'Esdeveniment': name,
                                            'Transició Principal': f"→ {states[max_idx]}",
                                            'Probabilitat': f"{max_prob:.3f}",
                                            'Descripció': {
                                                'TOV': 'Pèrdua → Acaba possessió',
                                                'Foul': 'Falta rebuda → Tirs lliures',
                                                'FT': 'Tir lliure → Acaba possessió',
                                                'Steal': 'Robada rival → Acaba possessió',
                                                'Block': 'Tap rival → Rebot',
                                                'Deflection': 'Deflexió → TOV o continua',
                                                'OREB': 'Rebot ofensiu → Nova oportunitat (START)',
                                                'DREB': 'Rebot defensiu → Acaba possessió'
                                            }[name]
                                        })
                                    
                                    events_df = pd.DataFrame(events_data)
                                    st.dataframe(events_df, use_container_width=True, hide_index=True)
                                    
                                    st.markdown("---")
                                    st.markdown("### 📊 Matriu Completa 17×17")
                                    st.dataframe(df_matrix, height=600, use_container_width=True)
                                
                                # Explicació de la matriu
                                st.markdown("---")
                                
                                st.markdown("""
                                
                                **Camins més comuns:**
                                
                                1. **TIR CONVERTIT** (el millor):
                                   ```
                                   START → Corner 3 → END
                                   (Tria tir) → (Converteix) → (3 punts!)
                                   ```
                                
                                2. **TIR FALLAT AMB REBOT DEFENSIU** (el pitjor):
                                   ```
                                   START → Layup → DREB → END
                                   (Entrada) → (Falla i rival agafa rebot) → (0 punts, perd possessió)
                                   ```
                                
                                3. **SEGONA OPORTUNITAT** (bo):
                                   ```
                                   START → Paint → OREB → START → Layup → END
                                   (Tir zona) → (Falla, rebot propi) → (Nova oportunitat) → (Converteix!)
                                   ```
                                
                                4. **PÈRDUA** (malament):
                                   ```
                                   START → TOV → END
                                   (Perd pilota directament) → (0 punts)
                                   ```
                                
                                5. **FALTA I TIRS LLIURES**:
                                   ```
                                   START → Dunk → Foul → FT → END
                                   (Intent esmaixada) → (Falta defensiva) → (Tir lliure) → (1-2 punts)
                                   ```
                                """)
                                
                                st.markdown("---")
                                st.info(f"""
                                📖 **Com interpretar la Matriu de Transició de {selected_player}:**
                                
                                Aquesta és una **Cadena de Markov de 17 estats** que modela el comportament individual del jugador.
                                
                                **📐 Lectura de la matriu:**
                                - **FILES (eix Y - vertical)**: Estat ACTUAL de la possessió
                                - **COLUMNES (eix X - horitzontal)**: Estat SEGÜENT (on pot transitar)
                                - **VALORS**: Probabilitat de transitar d'un estat a un altre
                                - **Propietat clau**: La suma de cada fila = 1.0 (100% de probabilitats)
                                
                                **🎯 Exemple pràctic - Fila "START":**
                                ```
                                START → Corner 3:    {player_matrix[0, 1]:.1%}  (probabilitat que tiri des del racó)
                                START → Above Break: {player_matrix[0, 2]:.1%}  (probabilitat que tiri frontal)
                                START → Layup:       {player_matrix[0, 6]:.1%}  (probabilitat que faci una entrada)
                                START → TOV:         {player_matrix[0, 8]:.1%}  (probabilitat de perdre la pilota)
                                ```
                                Això mostra la **distribució de tirs del jugador** - quines zones prefereix.
                                
                                **🏀 Exemple - Fila "Corner 3":**
                                ```
                                Corner 3 → END:  {player_matrix[1, 16]:.1%}  (probabilitat de CONVERTIR el tir)
                                Corner 3 → DREB: {player_matrix[1, 15]:.1%}  (probabilitat de fallar → rebot defensiu)
                                Corner 3 → OREB: {player_matrix[1, 14]:.1%}  (probabilitat de fallar → rebot ofensiu)
                                Corner 3 → Foul: {player_matrix[1, 9]:.1%}   (probabilitat de rebre falta)
                                ```
                                Això mostra què passa després d'intentar un tir de 3 des del racó.
                                
                                **📊 Els 17 Estats:**
                                - **START (0)**: Inici de possessió - distribució inicial
                                - **Zones de tir (1-7)**: Corner 3, Above Break 3, Mid-Range, Long Mid, Paint, Layup, Dunk
                                - **TOV (8)**: Pèrdua de pilota
                                - **Foul (9)**: Falta rebuda → va a FT
                                - **FT (10)**: Tir lliure
                                - **Steal (11)**: Robada defensiva (del rival)
                                - **Block (12)**: Tap defensiu (del rival)
                                - **Deflection (13)**: Deflexió (del rival)
                                - **OREB (14)**: Rebot ofensiu → torna a START
                                - **DREB (15)**: Rebot defensiu → END (perd possessió)
                                - **END (16)**: Final de possessió (estat absorbent)
                                
                                **🔬 Interpretació estadística:**
                                La matriu captura l'estil de joc del jugador basant-se en dades reals del shot chart.
                                Les files amb més verd indiquen accions més probables del jugador.
                                """)
                                
                                # Afegir exemple visual
                                st.markdown("---")
                                st.subheader("🔍 Exemple Visual de Lectura")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**📌 Fila START (inici possessió)**")
                                    start_probs = {
                                        'Corner 3': player_matrix[0, 1],
                                        'Above Break 3': player_matrix[0, 2],
                                        'Mid-Range': player_matrix[0, 3],
                                        'Long Mid': player_matrix[0, 4],
                                        'Paint': player_matrix[0, 5],
                                        'Layup': player_matrix[0, 6],
                                        'Dunk': player_matrix[0, 7],
                                        'TOV': player_matrix[0, 8]
                                    }
                                    
                                    # Ordenar per probabilitat
                                    sorted_starts = sorted(start_probs.items(), key=lambda x: x[1], reverse=True)
                                    
                                    st.markdown("**Zones preferides del jugador:**")
                                    for action, prob in sorted_starts[:5]:
                                        if prob > 0:
                                            st.write(f"• {action}: {prob:.1%}")
                                
                                with col2:
                                    st.markdown("**📌 Fila Corner 3 (després del tir)**")
                                    corner3_probs = {
                                        'Converteix (→END)': player_matrix[1, 16],
                                        'Falla (→DREB)': player_matrix[1, 15],
                                        'Rebot ofensiu (→OREB)': player_matrix[1, 14],
                                        'Falta (→Foul)': player_matrix[1, 9],
                                        'Tap (→Block)': player_matrix[1, 12]
                                    }
                                    
                                    sorted_corner3 = sorted(corner3_probs.items(), key=lambda x: x[1], reverse=True)
                                    
                                    st.markdown("**Resultats possibles:**")
                                    for outcome, prob in sorted_corner3:
                                        if prob > 0:
                                            st.write(f"• {outcome}: {prob:.1%}")
                                
                                # Mostrar estadístiques de tir - VISUAL COURT
                                st.markdown("---")
                                
                                # Foto del jugador + títol del shot chart amb color de l'equip
                                player_name = player_row.get('player_name', 'Jugador')
                                player_photo_html = get_player_photo_html(player_name, size=80, round_img=True)
                                player_team_color = get_team_primary_color(team_to_show)
                                
                                st.markdown(f"""
                                <div style='display: flex; align-items: center; gap: 1.5rem; padding: 1rem; background: white; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); margin: 1rem 0; border-left: 4px solid {player_team_color};'>
                                    {player_photo_html}
                                    <div>
                                        <div style='font-size: 1.5rem; font-weight: 700; color: #1A1A2E;'>🏀 Shot Chart</div>
                                        <div style='font-size: 1.1rem; color: {player_team_color}; font-weight: 600;'>{player_name}</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Obtenir percentatges
                                corner_3_pct = player_row.get('corner_3_pct', 0) * 100
                                above_break_3_pct = player_row.get('above_break_3_pct', 0) * 100
                                mid_range_pct = player_row.get('mid_range_pct', 0) * 100
                                long_mid_pct = player_row.get('long_mid_range_pct', 0) * 100
                                paint_pct = player_row.get('paint_pct', 0) * 100
                                layup_pct = player_row.get('layup_pct', 0) * 100
                                dunk_pct = player_row.get('dunk_pct', 0) * 100
                                
                                # Obtenir freqüències
                                corner_3_freq = player_row.get('corner_3_freq', 0) * 100
                                above_break_3_freq = player_row.get('above_break_3_freq', 0) * 100
                                mid_range_freq = player_row.get('mid_range_freq', 0) * 100
                                long_mid_freq = player_row.get('long_mid_range_freq', 0) * 100
                                paint_freq = player_row.get('paint_freq', 0) * 100
                                layup_freq = player_row.get('layup_freq', 0) * 100
                                dunk_freq = player_row.get('dunk_freq', 0) * 100
                                
                                # Funció per assignar color segons percentatge
                                def get_zone_color(pct):
                                    """Color segons percentatge"""
                                    if pct >= 50:
                                        return "#2ECC40"
                                    elif pct >= 40:
                                        return "#7FDB7F"
                                    elif pct >= 30:
                                        return "#FFDC00"
                                    elif pct >= 20:
                                        return "#FF851B"
                                    else:
                                        return "#FF4136"
                                
                                # Convertir imatge de pista a base64 per embedded HTML
                                import base64
                                import os
                                
                                # Buscar la imatge en diferents formats
                                court_image_b64 = None
                                court_image_mime = None
                                
                                image_paths = [
                                    ("imatges/pista_basquet.webp", "image/webp"),
                                    ("imatges/pista_basquet.png", "image/png"),
                                    ("imatges/pista_basquet.jpg", "image/jpeg"),
                                ]
                                
                                for path, mime in image_paths:
                                    if os.path.exists(path):
                                        with open(path, "rb") as img_file:
                                            court_image_b64 = base64.b64encode(img_file.read()).decode()
                                            court_image_mime = mime
                                        break
                                
                                # Si hi ha imatge, fer overlay damunt
                                if court_image_b64:
                                    # ========================================
                                    # COORDENADES EXACTES (imatge 283x197)
                                    # ========================================
                                    # CISTELLA: (142, 176), radi 3
                                    # PAINT: (108, 90) a (176, 197), 68×107
                                    # FREE THROW LINE: y=90, x=108 a 176
                                    # FREE THROW CIRCLE: centre (142, 90), radi 34
                                    # RESTRICTED AREA: centre (142, 176), radi 22
                                    # CORNER 3 LEFT: x=31, y=118-197
                                    # CORNER 3 RIGHT: x=252, y=118-197
                                    # THREE POINT ARC: centre (142, 176), radi 118
                                    # ZONES:
                                    #   - DUNK: cercle r=10 centre cistella
                                    #   - LAYUP: cercle r=28 centre cistella
                                    #   - SHORT MID: dins arc 3, y>120
                                    #   - LONG MID: dins arc 3, y<=120
                                    #   - ABOVE BREAK 3: fora arc, x entre 31-252
                                    #   - CORNER 3: x<31 o x>252
                                    # ========================================
                                    court_html = f"""
                                    <!DOCTYPE html>
                                    <html>
                                    <head>
                                        <style>
                                            body {{ margin: 0; padding: 0; background: white; font-family: 'Inter', sans-serif; }}
                                            .court-container {{ display: flex; justify-content: center; padding: 20px; background: white; }}
                                            .court-wrapper {{ position: relative; width: 700px; height: 487px; }}
                                            .court-image {{ width: 100%; height: 100%; object-fit: contain; }}
                                            .court-overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; }}
                                        </style>
                                    </head>
                                    <body>
                                        <div class="court-container">
                                            <div class="court-wrapper">
                                                <img class="court-image" src="data:{court_image_mime};base64,{court_image_b64}" alt="Pista NBA"/>
                                                <svg class="court-overlay" viewBox="0 0 283 197" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
                                                    
                                                    <!-- ============================ -->
                                                    <!-- CORNER 3 ESQUERRE (x < 31)   -->
                                                    <!-- ============================ -->
                                                    <rect x="0" y="120" width="17" height="77" 
                                                          fill="{get_zone_color(corner_3_pct)}" fill-opacity="0.55"/>
                                                    <text x="10" y="155" text-anchor="middle" fill="white" font-size="9" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.9);">{corner_3_pct:.0f}%</text>
                                                    <text x="10" y="165" text-anchor="middle" fill="white" font-size="5" style="text-shadow: 1px 1px 1px rgba(0,0,0,0.9);">Corner 3</text>
                                                    <text x="10" y="173" text-anchor="middle" fill="white" font-size="4" style="text-shadow: 1px 1px 1px rgba(0,0,0,0.9);">{corner_3_freq:.0f}%</text>
                                                    
                                                    <!-- ============================ -->
                                                    <!-- CORNER 3 DRET (x > 266)      -->
                                                    <!-- ============================ -->
                                                    <rect x="267" y="120" width="17" height="77" 
                                                          fill="{get_zone_color(corner_3_pct)}" fill-opacity="0.55"/>
                                                    <text x="273" y="155" text-anchor="middle" fill="white" font-size="9" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.9);">{corner_3_pct:.0f}%</text>
                                                    <text x="273" y="165" text-anchor="middle" fill="white" font-size="5" style="text-shadow: 1px 1px 1px rgba(0,0,0,0.9);">Corner 3</text>
                                                    <text x="273" y="173" text-anchor="middle" fill="white" font-size="4" style="text-shadow: 1px 1px 1px rgba(0,0,0,0.9);">{corner_3_freq:.0f}%</text>
                                                    
                                                    <!-- ============================ -->
                                                    <!-- ABOVE BREAK 3                 -->
                                                    <!-- Fora arc 3, des de extrem    -->
                                                    <!-- esquerre a extrem dret       -->
                                                    <!-- ============================ -->
                                                    <path d="M 17 120 A 133 133 0 0 1 266 120 L 283 0 L 0 0 L 17 120 Z"
                                                          fill="{get_zone_color(above_break_3_pct)}" fill-opacity="0.55"/>
                                                    <path d="M 283 0 L 266 120 L 283 120 Z"
                                                          fill="{get_zone_color(above_break_3_pct)}" fill-opacity="0.55"/>
                                                    <path d="M 0 0 L 17 120 L 0 120 Z"
                                                          fill="{get_zone_color(above_break_3_pct)}" fill-opacity="0.55"/>
                                                    <text x="142" y="15" text-anchor="middle" fill="white" font-size="14" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.9);">{above_break_3_pct:.0f}%</text>
                                                    <text x="142" y="22" text-anchor="middle" fill="white" font-size="7" style="text-shadow: 1px 1px 1px rgba(0,0,0,0.9);">Above Break 3</text>
                                                    <text x="142" y="29" text-anchor="middle" fill="white" font-size="6" style="text-shadow: 1px 1px 1px rgba(0,0,0,0.9);">Freq: {above_break_3_freq:.1f}%</text>
                                                    
                                                    <!-- ============================ -->
                                                    <!-- LONG MID-RANGE               -->
                                                    <!-- ============================ -->
                                                    
                                                    <path d="
                                                        M 17 120
                                                        A 133 133 0 0 1 266 120
                                                        L 266 120
                                                        L 176 120
                                                        L 176 90
                                                        L 108 90
                                                        L 108 120
                                                        L 17 120
                                                        Z"
                                                        fill="{get_zone_color(long_mid_pct)}"
                                                        fill-opacity="0.55"/>
                                                    
                                                    <text x="142" y="68" text-anchor="middle" fill="white" font-size="10" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.9);">Long Mid {long_mid_pct:.0f}%</text>
                                                    <text x="142" y="78" text-anchor="middle" fill="white" font-size="5" style="text-shadow: 1px 1px 1px rgba(0,0,0,0.9);">Freq: {long_mid_freq:.1f}%</text>
                                                    
                                                    <!-- ============================ -->
                                                    <!-- SHORT MID-RANGE               -->
                                                    <!-- Dins arc 3 + y > 120         -->
                                                    <!-- Laterals del paint            -->
                                                    <!-- ============================ -->
                                                    <!-- ESQUERRE -->
                                                    <path d="M 18 120 L 108 120 L 108 197 L 18 197 Z" 
                                                          fill="{get_zone_color(mid_range_pct)}" fill-opacity="0.55"/>
                                                    <text x="59" y="155" text-anchor="middle" fill="white" font-size="10" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.9);">{mid_range_pct:.0f}%</text>
                                                    <text x="59" y="167" text-anchor="middle" fill="white" font-size="6" style="text-shadow: 1px 1px 1px rgba(0,0,0,0.9);">Mid-Range</text>
                                                    <text x="59" y="177" text-anchor="middle" fill="white" font-size="5" style="text-shadow: 1px 1px 1px rgba(0,0,0,0.9);">{mid_range_freq:.1f}%</text>
                                                    
                                                    <!-- DRET -->
                                                    <path d="M 176 120 L 266 120 L 266 197 L 176 197 Z" 
                                                          fill="{get_zone_color(mid_range_pct)}" fill-opacity="0.55"/>
                                                    <text x="224" y="155" text-anchor="middle" fill="white" font-size="10" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.9);">{mid_range_pct:.0f}%</text>
                                                    <text x="224" y="167" text-anchor="middle" fill="white" font-size="6" style="text-shadow: 1px 1px 1px rgba(0,0,0,0.9);">Mid-Range</text>
                                                    <text x="224" y="177" text-anchor="middle" fill="white" font-size="5" style="text-shadow: 1px 1px 1px rgba(0,0,0,0.9);">{mid_range_freq:.1f}%</text>
                                                    
                                                    <!-- ============================ -->
                                                    <!-- PAINT                         -->
                                                    <!-- (108, 90) a (176, 197)       -->
                                                    <!-- ============================ -->
                                                    <rect x="108" y="90" width="68" height="107" 
                                                          fill="{get_zone_color(paint_pct)}" fill-opacity="0.45"/>
                                                    <text x="142" y="120" text-anchor="middle" fill="white" font-size="13" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.9);">{paint_pct:.0f}%</text>
                                                    <text x="142" y="132" text-anchor="middle" fill="white" font-size="7" style="text-shadow: 1px 1px 1px rgba(0,0,0,0.9);">Paint</text>
                                                    <text x="142" y="142" text-anchor="middle" fill="white" font-size="6" style="text-shadow: 1px 1px 1px rgba(0,0,0,0.9);">Freq: {paint_freq:.1f}%</text>
                                                    
                                                    <!-- ============================ -->
                                                    <!-- LAYUP ZONE                    -->
                                                    <!-- Cercle r=28 centre (142,176) -->
                                                    <!-- Només mig superior            -->
                                                    <!-- ============================ -->
                                                    <path d="M 114 176 A 28 28 0 0 1 170 176 L 170 197 L 114 197 Z" 
                                                          fill="{get_zone_color(layup_pct)}" fill-opacity="0.65"/>
                                                    <text x="142" y="170" text-anchor="middle" fill="white" font-size="8" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.9);">Layup: {layup_pct:.0f}%</text>
                                                    
                                                    <!-- ============================ -->
                                                    <!-- DUNK ZONE                     -->
                                                    <!-- Cercle r=10 centre (142,176) -->
                                                    <!-- ============================ -->
                                                    <circle cx="142" cy="176" r="10" 
                                                            fill="{get_zone_color(dunk_pct)}" fill-opacity="0.85"/>
                                                    <text x="142" y="178" text-anchor="middle" fill="white" font-size="6" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.9);">{dunk_pct:.0f}%</text>
                                                    <text x="142" y="186" text-anchor="middle" fill="white" font-size="4" style="text-shadow: 1px 1px 1px rgba(0,0,0,0.9);">Dunk</text>
                                                </svg>
                                            </div>
                                        </div>
                                    </body>
                                    </html>
                                    """
                                else:
                                    # Avís si no hi ha imatge
                                    st.info("💡 Per usar una imatge real, posa-la a `imatges/pista_basquet.png`")
                                    court_html = f"""
                                <!DOCTYPE html>
                                <html>
                                <head>
                                    <style>
                                        body {{ margin: 0; padding: 0; background: white; font-family: 'Inter', sans-serif; }}
                                        .court-container {{ 
                                            display: flex; 
                                            justify-content: center; 
                                            padding: 20px; 
                                            background: white;
                                        }}
                                    </style>
                                </head>
                                <body>
                                    <div class="court-container">
                                        <svg width="500" height="470" viewBox="0 0 500 470" xmlns="http://www.w3.org/2000/svg">
                                            <!-- DEFINICIONS - clipPath per limitar les zones a la pista -->
                                            <defs>
                                                <!-- Clip path per limitar les zones a la pista -->
                                                <clipPath id="court-clip">
                                                    <rect x="0" y="0" width="500" height="470"/>
                                                </clipPath>
                                                
                                                <!-- Clip per zona dins de línia de 3 -->
                                                <clipPath id="inside-three">
                                                    <path d="M 0 470 L 0 280 L 70 280 L 70 130 A 180 180 0 0 1 430 130 L 430 280 L 500 280 L 500 470 Z"/>
                                                </clipPath>
                                                
                                                <!-- Clip per zona fora de línia de 3 -->
                                                <clipPath id="outside-three">
                                                    <path d="M 0 0 L 500 0 L 500 280 L 430 280 L 430 130 A 180 180 0 0 0 70 130 L 70 280 L 0 280 Z"/>
                                                </clipPath>
                                            </defs>
                                            
                                            <!-- FONS DEL PARQUET -->
                                            <rect x="0" y="0" width="500" height="470" fill="#C19A6B"/>
                                            
                                            <!-- ============================================ -->
                                            <!-- ZONES DE TIR (sota les línies)              -->
                                            <!-- ============================================ -->
                                            
                                            <!-- Corner 3 ESQUERRE (entre línia base i línia de 3 paral·lela) -->
                                            <rect x="0" y="280" width="70" height="190" 
                                                  fill="{get_zone_color(corner_3_pct)}" fill-opacity="0.80"/>
                                            
                                            <!-- Corner 3 DRET -->
                                            <rect x="430" y="280" width="70" height="190" 
                                                  fill="{get_zone_color(corner_3_pct)}" fill-opacity="0.80"/>
                                            
                                            <!-- Above Break 3 (arc fora de la línia de 3) -->
                                            <path d="M 70 280 L 70 130 A 180 180 0 0 1 430 130 L 430 280 L 250 280 Z" 
                                                  fill="{get_zone_color(above_break_3_pct)}" fill-opacity="0.80"
                                                  clip-path="url(#outside-three)"/>
                                            <!-- Bandera per damunt: només la zona FORA de la línia de 3 -->
                                            <path d="M 70 280 L 70 30 L 430 30 L 430 280 L 250 280 A 180 180 0 0 0 70 280 Z" 
                                                  fill="{get_zone_color(above_break_3_pct)}" fill-opacity="0.80"/>
                                            
                                            <!-- Long Mid-Range (entre paint i línia de 3) -->
                                            <path d="M 70 310 L 70 280 A 180 180 0 0 1 430 280 L 430 310 L 325 310 L 325 280 L 175 280 L 175 310 Z" 
                                                  fill="{get_zone_color(long_mid_pct)}" fill-opacity="0.80"/>
                                            
                                            <!-- Mid-Range (al voltant del paint, fora de la zona pintada) -->
                                            <path d="M 70 310 L 70 380 L 175 380 L 175 310 Z" 
                                                  fill="{get_zone_color(mid_range_pct)}" fill-opacity="0.80"/>
                                            <path d="M 325 310 L 325 380 L 430 380 L 430 310 Z" 
                                                  fill="{get_zone_color(mid_range_pct)}" fill-opacity="0.80"/>
                                            <path d="M 70 380 L 70 470 L 175 470 L 175 380 Z" 
                                                  fill="{get_zone_color(mid_range_pct)}" fill-opacity="0.80"/>
                                            <path d="M 325 380 L 325 470 L 430 470 L 430 380 Z" 
                                                  fill="{get_zone_color(mid_range_pct)}" fill-opacity="0.80"/>
                                            
                                            <!-- Paint zone (la zona pintada/restricted) - exterior -->
                                            <rect x="175" y="280" width="150" height="120" 
                                                  fill="{get_zone_color(paint_pct)}" fill-opacity="0.80"/>
                                            
                                            <!-- Layup zone (semicercle restricted area) -->
                                            <path d="M 210 415 A 40 40 0 0 1 290 415 L 290 400 L 210 400 Z" 
                                                  fill="{get_zone_color(layup_pct)}" fill-opacity="0.85"/>
                                            
                                            <!-- Dunk zone (sota la cistella, molt prop) -->
                                            <circle cx="250" cy="415" r="20" 
                                                    fill="{get_zone_color(dunk_pct)}" fill-opacity="0.95"/>
                                            
                                            <!-- ============================================ -->
                                            <!-- LÍNIES DE LA PISTA (sobre les zones)        -->
                                            <!-- ============================================ -->
                                            
                                            <!-- Línia base (sota) -->
                                            <line x1="0" y1="470" x2="500" y2="470" stroke="white" stroke-width="3"/>
                                            
                                            <!-- Línies laterals -->
                                            <line x1="0" y1="0" x2="0" y2="470" stroke="white" stroke-width="3"/>
                                            <line x1="500" y1="0" x2="500" y2="470" stroke="white" stroke-width="3"/>
                                            
                                            <!-- Línia mig camp (a dalt) -->
                                            <line x1="0" y1="0" x2="500" y2="0" stroke="white" stroke-width="3"/>
                                            
                                            <!-- LÍNIA DE 3 PUNTS -->
                                            <!-- Parts paral·leles dels corners (rectes) -->
                                            <line x1="70" y1="280" x2="70" y2="470" stroke="white" stroke-width="3"/>
                                            <line x1="430" y1="280" x2="430" y2="470" stroke="white" stroke-width="3"/>
                                            <!-- Arc superior de la línia de 3 -->
                                            <path d="M 70 280 A 180 180 0 0 1 430 280" fill="none" stroke="white" stroke-width="3"/>
                                            
                                            <!-- PAINT (zona pintada / restricted area) -->
                                            <rect x="175" y="280" width="150" height="190" 
                                                  fill="none" stroke="white" stroke-width="3"/>
                                            
                                            <!-- Free throw line (línia de tirs lliures) -->
                                            <line x1="175" y1="280" x2="325" y2="280" stroke="white" stroke-width="3"/>
                                            
                                            <!-- Free throw circle (cercle de tirs lliures) -->
                                            <!-- Mitja part superior contínua -->
                                            <path d="M 190 280 A 60 60 0 0 1 310 280" fill="none" stroke="white" stroke-width="2"/>
                                            <!-- Mitja part inferior amb dashes -->
                                            <path d="M 190 280 A 60 60 0 0 0 310 280" fill="none" stroke="white" stroke-width="2" stroke-dasharray="8,5"/>
                                            
                                            <!-- Restricted area (semicercle prop de la cistella) -->
                                            <path d="M 210 415 A 40 40 0 0 1 290 415" fill="none" stroke="white" stroke-width="2"/>
                                            
                                            <!-- Backboard (taulell) -->
                                            <line x1="220" y1="425" x2="280" y2="425" stroke="white" stroke-width="5"/>
                                            
                                            <!-- Aro de la cistella -->
                                            <circle cx="250" cy="415" r="9" fill="none" stroke="#E94B0C" stroke-width="3"/>
                                            <!-- Suport de l'aro al taulell -->
                                            <line x1="250" y1="425" x2="250" y2="420" stroke="white" stroke-width="2"/>
                                            
                                            <!-- ============================================ -->
                                            <!-- TEXT AMB PERCENTATGES                       -->
                                            <!-- ============================================ -->
                                            
                                            <!-- Corner 3 esquerre -->
                                            <text x="35" y="365" text-anchor="middle" fill="white" font-size="20" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{corner_3_pct:.0f}%</text>
                                            <text x="35" y="385" text-anchor="middle" fill="white" font-size="10" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">Corner 3</text>
                                            <text x="35" y="400" text-anchor="middle" fill="white" font-size="9" opacity="0.9" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{corner_3_freq:.1f}%</text>
                                            
                                            <!-- Corner 3 dret -->
                                            <text x="465" y="365" text-anchor="middle" fill="white" font-size="20" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{corner_3_pct:.0f}%</text>
                                            <text x="465" y="385" text-anchor="middle" fill="white" font-size="10" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">Corner 3</text>
                                            <text x="465" y="400" text-anchor="middle" fill="white" font-size="9" opacity="0.9" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{corner_3_freq:.1f}%</text>
                                            
                                            <!-- Above Break 3 -->
                                            <text x="250" y="90" text-anchor="middle" fill="white" font-size="22" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{above_break_3_pct:.0f}%</text>
                                            <text x="250" y="110" text-anchor="middle" fill="white" font-size="12" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">Above Break 3</text>
                                            <text x="250" y="128" text-anchor="middle" fill="white" font-size="10" opacity="0.9" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{above_break_3_freq:.1f}%</text>
                                            
                                            <!-- Long Mid-Range -->
                                            <text x="250" y="225" text-anchor="middle" fill="white" font-size="14" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">Long Mid {long_mid_pct:.0f}%</text>
                                            <text x="250" y="240" text-anchor="middle" fill="white" font-size="9" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{long_mid_freq:.1f}%</text>
                                            
                                            <!-- Mid-Range esquerra -->
                                            <text x="122" y="420" text-anchor="middle" fill="white" font-size="14" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{mid_range_pct:.0f}%</text>
                                            <text x="122" y="438" text-anchor="middle" fill="white" font-size="9" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">Mid-Range</text>
                                            <text x="122" y="452" text-anchor="middle" fill="white" font-size="8" opacity="0.9" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{mid_range_freq:.1f}%</text>
                                            
                                            <!-- Mid-Range dreta -->
                                            <text x="378" y="420" text-anchor="middle" fill="white" font-size="14" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{mid_range_pct:.0f}%</text>
                                            <text x="378" y="438" text-anchor="middle" fill="white" font-size="9" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">Mid-Range</text>
                                            <text x="378" y="452" text-anchor="middle" fill="white" font-size="8" opacity="0.9" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{mid_range_freq:.1f}%</text>
                                            
                                            <!-- Paint -->
                                            <text x="250" y="335" text-anchor="middle" fill="white" font-size="18" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{paint_pct:.0f}%</text>
                                            <text x="250" y="353" text-anchor="middle" fill="white" font-size="11" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">Paint</text>
                                            <text x="250" y="370" text-anchor="middle" fill="white" font-size="9" opacity="0.9" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{paint_freq:.1f}%</text>
                                            
                                            <!-- Layup -->
                                            <text x="250" y="383" text-anchor="middle" fill="white" font-size="10" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">Layup: {layup_pct:.0f}% ({layup_freq:.1f}%)</text>
                                            
                                            <!-- Dunk (sota cistella) -->
                                            <text x="250" y="415" text-anchor="middle" fill="white" font-size="11" font-weight="bold" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">{dunk_pct:.0f}%</text>
                                            <text x="250" y="430" text-anchor="middle" fill="white" font-size="8" style="text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">Dunk</text>
                                        </svg>
                                    </div>
                                </body>
                                </html>
                                """
                                
                                # Renderitzar amb components.html (no st.markdown)
                                import streamlit.components.v1 as components
                                components.html(court_html, height=540, scrolling=False)
                                
                                # Llegenda de colors
                                st.markdown("""
                                <div style='display: flex; justify-content: center; gap: 1.5rem; padding: 1rem; background: white; border-radius: 12px; margin-top: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.08); flex-wrap: wrap;'>
                                    <div style='display: flex; align-items: center; gap: 0.5rem;'>
                                        <div style='width: 20px; height: 20px; background: #2ECC40; border-radius: 4px;'></div>
                                        <span style='font-size: 0.85rem; font-weight: 500;'>≥ 50%</span>
                                    </div>
                                    <div style='display: flex; align-items: center; gap: 0.5rem;'>
                                        <div style='width: 20px; height: 20px; background: #7FDB7F; border-radius: 4px;'></div>
                                        <span style='font-size: 0.85rem; font-weight: 500;'>40-50%</span>
                                    </div>
                                    <div style='display: flex; align-items: center; gap: 0.5rem;'>
                                        <div style='width: 20px; height: 20px; background: #FFDC00; border-radius: 4px;'></div>
                                        <span style='font-size: 0.85rem; font-weight: 500;'>30-40%</span>
                                    </div>
                                    <div style='display: flex; align-items: center; gap: 0.5rem;'>
                                        <div style='width: 20px; height: 20px; background: #FF851B; border-radius: 4px;'></div>
                                        <span style='font-size: 0.85rem; font-weight: 500;'>20-30%</span>
                                    </div>
                                    <div style='display: flex; align-items: center; gap: 0.5rem;'>
                                        <div style='width: 20px; height: 20px; background: #FF4136; border-radius: 4px;'></div>
                                        <span style='font-size: 0.85rem; font-weight: 500;'>&lt; 20%</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Taula resum al final (opcional)
                                with st.expander("📋 Veure dades detallades en taula"):
                                    shooting_data = {
                                        'Zona': [
                                            '🎯 Corner 3',
                                            '🎯 Above Break 3',
                                            '🎯 Mid-Range',
                                            '🎯 Long Mid',
                                            '🎨 Paint',
                                            '🏃 Layup',
                                            '💪 Dunk'
                                        ],
                                        '% Conversió': [
                                            f"{corner_3_pct:.1f}%",
                                            f"{above_break_3_pct:.1f}%",
                                            f"{mid_range_pct:.1f}%",
                                            f"{long_mid_pct:.1f}%",
                                            f"{paint_pct:.1f}%",
                                            f"{layup_pct:.1f}%",
                                            f"{dunk_pct:.1f}%"
                                        ],
                                        '% Freqüència': [
                                            f"{corner_3_freq:.1f}%",
                                            f"{above_break_3_freq:.1f}%",
                                            f"{mid_range_freq:.1f}%",
                                            f"{long_mid_freq:.1f}%",
                                            f"{paint_freq:.1f}%",
                                            f"{layup_freq:.1f}%",
                                            f"{dunk_freq:.1f}%"
                                        ]
                                    }
                                    df_shooting = pd.DataFrame(shooting_data)
                                    st.dataframe(df_shooting, use_container_width=True, hide_index=True)
                                
                            except Exception as e:
                                st.error(f"❌ Error generant matriu: {e}")
                                import traceback
                                st.code(traceback.format_exc())
                
            except Exception as e:
                st.error(f"❌ Error carregant matrius: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Footer professional
    st.markdown("""
    <div class="footer">
        <div style='display: flex; justify-content: center; align-items: center; gap: 2rem; flex-wrap: wrap;'>
            <div>
                <strong style='color: #17408B;'>🏀 NBA Game Predictor</strong>
            </div>
            <div style='color: #6C757D;'>·</div>
            <div>
                Treball Fi de Grau · Matemàtica Computacional
            </div>
            <div style='color: #6C757D;'>·</div>
            <div>
                Universitat Autònoma de Barcelona · 2025
            </div>
        </div>
        <div style='margin-top: 0.5rem; font-size: 0.75rem; color: #adb5bd;'>
            Sistema basat en cadenes de Markov i simulació Monte Carlo
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
