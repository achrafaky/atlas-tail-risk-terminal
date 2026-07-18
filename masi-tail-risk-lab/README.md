<div align="center">

# 📊 ATLAS — Tail Risk Terminal

### *Institutional-grade tail risk & margin engine for the MASI Index*

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/CI-passing-brightgreen?style=for-the-badge&logo=githubactions&logoColor=white" />
</p>

<p>
  <img src="https://img.shields.io/badge/GEV-EVT-0057B7?style=flat-square" />
  <img src="https://img.shields.io/badge/POT-Peaks%20Over%20Threshold-0057B7?style=flat-square" />
  <img src="https://img.shields.io/badge/GARCH--EVT-Conditional%20Volatility-0057B7?style=flat-square" />
  <img src="https://img.shields.io/badge/HMM-Regime%20Detection-0057B7?style=flat-square" />
</p>

**Master MMSD — FST Tanger** · Achraf Akiyaf · Supervisor: Prof. AZMANI Abdellah

</div>

<br>

<div align="center">
  <img src="report/figures/dashboard_overview.png" width="90%" alt="ATLAS Dashboard Overview">
</div>

<br>

---

## ✨ Overview

**ATLAS** is a real-time tail-risk and margin desk built on the **MASI 20** index (Casablanca Stock Exchange). It goes beyond textbook VaR by stacking four risk models of increasing sophistication — Gaussian, Historical, **Extreme Value Theory (POT)**, and **GARCH-EVT** — to show *why* naive models systematically underestimate crash risk, and to compute the capital a trading desk should actually hold.

> Built as a pedagogical + quantitative research tool, not a trading signal generator.

---

## 🖥️ Live Dashboard

<table>
<tr>
<td width="50%">

**Risk Gauge & Model Comparison**

<img src="report/figures/risk_gauge_models.png" width="100%">

</td>
<td width="50%">

**Drawdown & Stress Periods**

<img src="report/figures/drawdown_stress.png" width="100%">

</td>
</tr>
</table>

---

## 🧠 Methodology

| Layer | Model | Purpose |
|---|---|---|
| **1. Volatility** | GARCH(1,1) | Captures volatility clustering & conditional heteroscedasticity |
| **2. Tail estimation** | GEV / POT (Peaks Over Threshold) | Models the fat tails that Gaussian VaR misses |
| **3. Combined** | GARCH-EVT | Conditional volatility × extreme value tail → most conservative, most realistic VaR |
| **4. Regimes** | Hidden Markov Model (HMM) | Detects latent market states (calm / turbulent / crash) |

**Result — VaR 99% (1-day) across models:**

| Model | VaR 99% |
|---|---|
| Gaussian | 1.70% |
| Historical | 1.92% |
| EVT (POT) | 2.04% |
| **GARCH-EVT** | **2.26%** |

The clear escalation demonstrates the core thesis: *ignoring fat tails and volatility clustering leads to structurally under-capitalized trading desks.*

---

## 🚀 Features

- 📈 **Live VaR 99% (EVT)** and recommended capital per contract
- 🔥 **"Replay worst day"** — one-click simulation of the 16/03/2020 crash (−9.23%)
- 🎚️ **What-If stress testing** — volatility shock slider recalculating VaR, potential loss, and required margin in real time
- 🎲 **Position simulator** — margin requirement for LONG/SHORT positions (illustrative, not a trading signal)
- 📉 **Drawdown tracker** — cumulative loss since last peak, worst historical drawdown: **−37.6%**
- 🧩 **Regime detection (HMM)** — market state classification
- 📄 **One-click PDF risk report generation**

---

## 🗂️ Project Structure

```
masi-tail-risk-lab/
├── .github/workflows/ci.yml        # CI pipeline (pytest on push/PR)
├── app/
│   └── dashboard.py                 # Streamlit application
├── data/
│   ├── raw/                         # Raw MASI historical data
│   └── processed/                   # Cleaned & feature-engineered data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_evt_modeling.ipynb        # GEV / POT fitting
│   ├── 03_margin_model.ipynb        # Capital / margin engine
│   ├── 04_backtesting.ipynb
│   └── 05_conditional_margin_stress.ipynb
├── src/
│   ├── __init__.py
│   └── backtest.py
├── report/
│   ├── figures/
│   └── RESULTATS_CLES.md
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/achrafaky/atlas-tail-risk-terminal.git
cd atlas-tail-risk-terminal
pip install -r requirements.txt
```

## ▶️ Run

```bash
streamlit run app/dashboard.py
```

The app will be available at `http://localhost:8501`.

---

## 🧪 Testing

```bash
pytest -q
```

CI runs automatically on every push / pull request via GitHub Actions (`.github/workflows/ci.yml`).

---

## 📚 Key Results

Full quantitative results, model diagnostics, and backtesting statistics are documented in [`report/RESULTATS_CLES.md`](report/RESULTATS_CLES.md).

---

## ⚠️ Disclaimer

This project is for **academic and research purposes only**. The position simulator draws randomly from the historical distribution and does **not** constitute financial or trading advice.

---

<div align="center">

### 👤 Author

**Achraf Akiyaf** — Master MMSD, FST Tanger
Supervised by Prof. AZMANI Abdellah

<sub>Built with Python, Streamlit, GARCH, EVT, and a healthy respect for fat tails.</sub>

</div>
