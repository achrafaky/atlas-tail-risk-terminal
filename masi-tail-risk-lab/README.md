<div align="center">

# ATLAS
### *Tail Risk & Margin Terminal — MASI 20 Index*

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
  <img src="https://img.shields.io/badge/Monte%20Carlo-Simulation-0057B7?style=flat-square" />
</p>

**Master MMSD — FST Tanger** · Achraf Akiyaf · Supervisor: Prof. AZMANI Abdellah

<br>

<img src="report/figures/dashboard%20atlas%201.png" width="100%" alt="ATLAS Dashboard">

</div>

<br>

## 📖 Table of Contents

- [Overview](#-overview)
- [Live Terminal](#️-live-terminal)
- [Methodology](#-methodology)
- [EVT Diagnostics](#-evt-diagnostics)
- [Risk Engine Results](#-risk-engine-results)
- [Stress Testing & Simulation](#️-stress-testing--simulation)
- [Project Structure](#️-project-structure)
- [Installation](#️-installation)
- [Testing](#-testing)
- [Disclaimer](#️-disclaimer)

---

## ✨ Overview

**ATLAS** is a real-time tail-risk and margin desk built on the **MASI 20** index (Casablanca Stock Exchange). It goes beyond textbook VaR by stacking four risk models of increasing sophistication — Gaussian, Historical, **Extreme Value Theory (POT)**, and **GARCH-EVT** — to demonstrate *why* naive models systematically underestimate crash risk, and to compute the capital a trading desk should actually hold against it.

Every model in the pipeline is backed by full statistical diagnostics (QQ-plots, mean residual life, parameter stability, block maxima, Monte Carlo loss distributions) — not just headline numbers.

> Built as a quantitative research & pedagogical tool — not a trading signal generator.

---

## 🖥️ Live Terminal

<table>
<tr>
<td width="50%">

**KPI Header — real-time risk metrics**
<img src="report/figures/dashboard%20atlas%205.png" width="100%">

</td>
<td width="50%">

**Market Chart — Price, Bollinger, VaR band**
<img src="report/figures/dashboard%20atlas%204.png" width="100%">

</td>
</tr>
<tr>
<td width="50%">

**Risk Gauge & Model Comparison**
<img src="report/figures/dashboard%20atlas%203.png" width="100%">

</td>
<td width="50%">

**Drawdown & Stress Periods**
<img src="report/figures/dashboard%20atlas%202.png" width="100%">

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
| **5. Validation** | Monte Carlo simulation | Simulates thousands of loss paths from the fitted distribution |

---

## 📊 EVT Diagnostics

Full diagnostic suite validating the Extreme Value Theory fit before it feeds the margin engine:

<table>
<tr>
<td width="33%" align="center">

**QQ-Plot**
<img src="report/figures/qq_plot.png" width="100%">
<sub>Goodness-of-fit of the GPD tail</sub>

</td>
<td width="33%" align="center">

**Mean Residual Life**
<img src="report/figures/mean%20residual%20life%20and%20parameter%20stability.png" width="100%">
<sub>Threshold selection for POT</sub>

</td>
<td width="33%" align="center">

**Block Maxima**
<img src="report/figures/bloc_maxima.png" width="100%">
<sub>GEV fit on annual/monthly maxima</sub>

</td>
</tr>
<tr>
<td width="33%" align="center">

**Return Level Plot**
<img src="report/figures/return%20level%20plot.png" width="100%">
<sub>Expected extreme return by return period</sub>

</td>
<td width="33%" align="center">

**Parameter Stability**
<img src="report/figures/mean%20residual%20life%20and%20parameter%20stability.png" width="100%">
<sub>Shape/scale stability across thresholds</sub>

</td>
<td width="33%" align="center">

**Monte Carlo Loss Distribution**
<img src="report/figures/Distribution%20Monte%20carlo%20des%20pertes.png" width="100%">
<sub>10,000+ simulated loss paths</sub>

</td>
</tr>
</table>

---

## 📈 Risk Engine Results

**VaR 99% (1-day) across models:**

| Model | VaR 99% |
|---|---|
| Gaussian | 1.70% |
| Historical | 1.92% |
| EVT (POT) | 2.04% |
| **GARCH-EVT** | **2.26%** |

The clear escalation demonstrates the core thesis: *ignoring fat tails and volatility clustering leads to structurally under-capitalized trading desks.*

<div align="center">
<img src="report/figures/Var%2099%20cond.png" width="70%" alt="Conditional VaR 99%">
</div>

**Key figures:**
- 📉 Worst historical drawdown: **−37.6%**
- 🔥 Worst historical day (16/03/2020, COVID crash): **−9.23%**
- 💰 Recommended capital per contract: **8,652 MAD**

---

## 🎚️ Stress Testing & Simulation

<img src="report/figures/stress%20tests%20de%20la%20marge.png" width="100%" alt="What-If stress test and margin simulator">

- **"Replay worst day"** — one-click simulation of the 16/03/2020 crash
- **What-If stress slider** — recalculates VaR, potential loss, and required margin under a custom volatility shock
- **Position simulator** — margin requirement for LONG/SHORT positions (illustrative, draws randomly from the historical distribution — not a trading signal)

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
│   ├── 02_evt_modeling.ipynb        # GEV / POT fitting, diagnostics
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

<sub>Built with Python, Streamlit, GARCH, EVT, HMM, and a healthy respect for fat tails.</sub>

</div>
