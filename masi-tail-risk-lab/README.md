# 🏔️ ATLAS Risk Terminal
### MASI Tail-Risk & Margin Desk

![Python](https://img.shields.io/badge/Python-3.11-blue)
![CI](https://img.shields.io/badge/tests-pytest-brightgreen)
![Status](https://img.shields.io/badge/status-work%20in%20progress-F0B90B)
![Market](https://img.shields.io/badge/market-Casablanca%20Stock%20Exchange-006233)
![Method](https://img.shields.io/badge/method-Extreme%20Value%20Theory-8A2BE2)

> **Quelle marge initiale une chambre de compensation devrait-elle exiger sur
> le Future MASI 20 ?** Calibration et backtesting de modèles de marge par la
> Statistique des Valeurs Extrêmes, sur données réelles de la Bourse de
> Casablanca — à l'occasion du lancement du marché à terme marocain
> (avril 2026).

**Auteur :** Achraf Akiyaf — Master MMSD (Mathématiques, Modélisation et
Sciences des Données), FST Tanger. *Projet réalisé dans une optique de stage
en gestion quantitative des risques.*

---

## 🎯 La question

Le 6 avril 2026, la Bourse de Casablanca a lancé son marché à terme avec un
premier contrat future sur l'indice MASI 20 (10 MAD par point d'indice) et
une chambre de compensation centrale. Toute CCP doit répondre à une question
fondamentale : **quelle marge initiale couvre les pertes extrêmes sur la
période de liquidation, sans étrangler le marché en temps de crise ?**

Ce projet y répond en comparant trois familles de modèles de marge
(gaussienne, historique filtrée, **EVT**) calibrées et backtestées sur
15 ans d'historique du MASI — y compris le krach de mars 2020.

## 🗺️ Roadmap (suivie en public)

- [ ] **Semaine 1 — Données & diagnostic** : nettoyage documenté de
      l'historique MASI, faits stylisés (queues épaisses, clustering de
      volatilité), échec de la VaR gaussienne démontré par backtest.
- [ ] **Semaine 2 — Moteur EVT** : GEV bloc-maxima, POT-GPD (choix du seuil
      par Mean Residual Life & Parameter Stability), VaR/ES fermées,
      GARCH-EVT conditionnel (McNeil & Frey, 2000).
- [ ] **Semaine 3 — Modèles de marge** : IM = VaR(MPOR 2j, 99–99.7%) ×
      indice × 10 MAD ; analyse de suffisance (mars 2020) et de
      **procyclicité** ; comparaison au dépôt de référence.
- [ ] **Semaine 4 — Validation & rapport** : backtesting walk-forward
      (Kupiec, Christoffersen), cas de couverture d'un portefeuille OPCVM,
      rapport PDF de recherche (FR), mini-dashboard Streamlit.

## 📁 Structure

```
masi-tail-risk-lab/
├── config.yaml            # paramètres centralisés (rien de hardcodé)
├── data/
│   ├── raw/               # CSV bruts (non versionnés — source : Bourse de
│   └── processed/         #   Casablanca / Investing.com)
├── notebooks/             # 01_data → 02_evt → 03_margin → 04_backtesting
├── src/                   # fonctions réutilisables et testées
├── tests/                 # pytest (CI GitHub Actions)
└── report/                # rapport PDF + figures
```

## 🔬 Méthodes

Théorie des Valeurs Extrêmes (GEV, GPD/POT), GJR-GARCH(1,1)-EVT,
bootstrap paramétrique, backtesting réglementaire (Kupiec POF,
Christoffersen conditional coverage), marges CCP (MPOR, anti-procyclicité).

**Références :** Coles (2001) ; McNeil & Frey (2000) ; McNeil, Frey &
Embrechts, *Quantitative Risk Management* ; standards CPMI-IOSCO sur les
marges des contreparties centrales.


## 🖥️ Dashboard "Trading Desk" (bonus)

Un dashboard Streamlit thème Binance, branché sur le moteur EVT réel :
métriques temps réel (spot, volatilité GARCH, VaR EVT en points, capital
recommandé), chandeliers + Bandes de Bollinger + bande VaR dynamique, jauge
de risque (RISK-ON / CAUTION / CRASH WARNING), comparaison des 4 modèles de
VaR, simulateur de position pédagogique, et graphique de drawdown avec
périodes de stress.

```bash
streamlit run app/dashboard.py
```

Nouvelles fonctionnalités : sélecteur de timeframe (1M→ALL), annotations automatiques des crises (COVID, Ukraine), simulateur "rejouer le pire jour historique", et **export PDF one-click** du rapport de risque (`src/report_pdf.py`).


## 🧠 Détection de régimes (HMM)

Un modèle de Markov caché (`src/regimes.py`) identifie trois régimes latents
de marché par leur volatilité — **Calme, Normal, Crise**. Sur le MASI, le
régime "Crise" (volatilité ~3× supérieure) capture correctement mars 2020.
Le dashboard colore le graphique de prix par régime. *Le HMM caractérise le
régime courant ; il ne prédit pas l'avenir.*


## 🏛️ Extensions "desk de risque CCP"

- **Marge conditionnelle GARCH-EVT** (`margin.garch_evt_margin`) : tient
  compte de la volatilité actuelle → moins procyclique (exigence des régulateurs).
- **Stress tests par scénarios** (`margin.stress_margin`) : choc de vol +100 %,
  flash crash, recalibrage COVID 2020. *Un scénario COVID triplerait la marge.*
- **Marge Monte Carlo** (`margin.monte_carlo_margin`) : 10 000 trajectoires
  semi-paramétriques (corps bootstrap + queue GPD), validant la formule
  analytique à moins de 1 %.

Voir `notebooks/05_conditional_margin_stress.ipynb`.


## 🏢 ATLAS Risk Terminal — Style institutionnel

Rebrand "Corporate Blue & Dark" (style Bloomberg/Refinitiv) : palette bleu
nuit `#0A1628`, typographie Inter + JetBrains Mono, cartes KPI avec
**sparklines** intégrées. Nouvelles fonctionnalités 100% basées sur les
données réelles :
- **Flash Alert** automatique quand la VaR dépasse 2× sa moyenne historique.
- **Stress Test "What-If"** : slider de choc de volatilité (0-200%) qui
  recalcule la VaR et la marge en temps réel.
- **Export PNG** des graphiques (nécessite `kaleido` + Chrome installé).


## 🎓 Analyses "Senior Quant / Head of Risk"

Six analyses avancées qui vont au-delà de la VaR statique :
- **MPOR Scaling** : projection de la VaR selon l'horizon de liquidation,
  via la vraie loi de puissance EVT `VaR(h) = VaR(1j) × h^ξ` (Danielsson &
  de Vries) — **pas** la racine du temps, valable seulement sous normalité.
- **Feu tricolore de Bâle** : zones officielles (0-4 / 5-9 / ≥10 violations
  sur 250 jours) intégrées au backtest personnalisé.
- **Sévérité des violations** : ratio Perte/VaR pour quantifier l'ampleur
  des échecs du modèle, pas seulement leur nombre.
- **Reverse Stress Test** : quelle perte % ferait exactement égaler la
  marge actuelle — teste la résilience de la CCP.
- **Sensibilité du seuil GPD** : la VaR varie de moins de 1 % sur ±20 % de
  changement du seuil → modèle jugé stable.
- **Backtest walk-forward** : p-value de Christoffersen sur fenêtres
  glissantes de 250 jours, pour vérifier la stabilité dans le temps.

## ⚠️ Disclaimer

Projet académique et pédagogique. Les paramètres du contrat cités
proviennent de sources publiques ; les résultats ne constituent ni un audit
de la CCP marocaine ni un conseil d'investissement.
