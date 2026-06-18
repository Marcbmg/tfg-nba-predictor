# 🏀 NBA Predictor - TFG

Aplicació web per a la predicció de partits NBA basada en cadenes
de Markov i simulació Monte Carlo. Treball Final de Grau en
Matemàtica Computacional, Universitat Autònoma de Barcelona.

## Demo en línia

🌐 Accedeix a l'aplicació desplegada a Streamlit Community Cloud
(URL pendent del desplegament).

## Característiques principals

- Predicció de marcadors NBA amb intervals de confiança
- Simulació Monte Carlo amb 1.000 iteracions per partit
- Anàlisi de quintets personalitzats
- Visualització de matrius de transició Markov
- Comparació de jugadors per zones de tir

## Model matemàtic

L'aplicació utilitza cadenes de Markov absorbents amb 17 estats
calibrades mitjançant estimació Bayesiana amb prior d'equip,
impacte defensiu bidireccional i avantatge de pista integrat.

## Tecnologies

- Python 3.10
- Streamlit
- NumPy, Pandas
- Plotly
- NBA Stats API

## Autor

Marc Vega Velilla - TFG, UAB 2026
