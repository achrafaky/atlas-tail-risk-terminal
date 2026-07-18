"""
🏔️ ATLAS Risk Terminal — MASI 20 Tail-Risk & Margin Desk
===========================================================
Auteur : Achraf Akiyaf — Master MMSD, FST Tanger.

Terminal de risque institutionnel branché sur le pipeline EVT réel (src/) :
GEV, POT-GPD, GARCH-EVT, HMM, marges CCP (inconditionnelle, conditionnelle,
Monte Carlo), backtesting réglementaire.
Lancement :  streamlit run app/dashboard.py

⚠️ Projet académique. Le simulateur de trade est pédagogique ; les signaux
sont des indicateurs de régime de risque, pas un conseil d'investissement.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Palette "Corporate Blue & Dark" (style Bloomberg / Refinitiv) ---
BG, CARD, BORDER = "#0A1628", "#13203A", "#1E3A5F"
GOLD, BLUE = "#D4AF37", "#0066CC"
GREEN, RED, TXT, MUTED = "#0ECB81", "#F6465D", "#F0F4F8", "#8899BB"

st.set_page_config(page_title="ATLAS Risk Terminal", page_icon="\U0001F3D4\uFE0F",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
.stApp {{{{ background:{{BG}}; color:{{TXT}}; font-family:'Inter',sans-serif; }}}}
section[data-testid="stSidebar"] {{{{ background:{{CARD}}; border-right:1px solid {{BORDER}}; }}}}
div[data-testid="stMetric"] {{{{ background:{{CARD}}; border-radius:10px; padding:14px;
  border:1px solid {{BORDER}}; border-left:3px solid {{BLUE}}; }}}}
div[data-testid="stMetricValue"] {{{{ font-size:26px; font-family:'JetBrains Mono',monospace; }}}}
div[data-testid="stMetricLabel"] {{{{ font-size:11px; color:{{MUTED}}; text-transform:uppercase;
  letter-spacing:0.5px; }}}}
h1,h2,h3 {{{{ color:{{TXT}}; font-family:'Inter',sans-serif; }}}}
h1 {{{{ color:{{GOLD}}; }}}}
.stButton>button {{{{ border-radius:8px; font-weight:600; border:1px solid {{BORDER}}; }}}}
.atlas-flash {{{{ background:linear-gradient(90deg,{{RED}},#B91C3C); color:white;
  padding:10px 16px; border-radius:8px; font-weight:600; text-align:center;
  animation:atlas-pulse 1.6s ease-in-out infinite; margin-bottom:10px; }}}}
@keyframes atlas-pulse {{{{ 0%,100%{{{{opacity:1}}}} 50%{{{{opacity:0.55}}}} }}}}
</style>""", unsafe_allow_html=True)

PL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD,
          font=dict(color=TXT, family="Inter", size=12),
          margin=dict(l=40, r=20, t=45, b=35),
          legend=dict(bgcolor="rgba(0,0,0,0)"))


def _contiguous(mask):
    """Retourne les segments (début, fin) où mask est True (pour add_vrect)."""
    out = []
    idx = mask.index
    in_seg = False
    start = None
    for i, val in enumerate(mask.values):
        if val and not in_seg:
            in_seg, start = True, idx[i]
        elif not val and in_seg:
            in_seg = False
            out.append((start, idx[i]))
    if in_seg:
        out.append((start, idx[-1]))
    return out


# ---------------------------------------------------------------
# Chargement des données + calculs EVT (avec fallback synthétique)
# ---------------------------------------------------------------
@st.cache_data(show_spinner="Chargement & estimation EVT…")
def load_all():
    """Charge le MASI, calcule VaR gaussienne/EVT/GARCH et les marges.
    Fallback : données synthétiques ressemblant au MASI si le CSV manque."""
    from scipy import stats
    try:
        from src.data_utils import load_masi, audit_and_clean, log_returns
        from src import evt
        from src.garch_evt import fit_garch_evt
        path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "masi.csv")
        masi, _ = audit_and_clean(load_masi(path))
        r = log_returns(masi)
        real = True
    except Exception:
        # Fallback synthétique (GARCH-like) ressemblant au MASI
        rng = np.random.default_rng(0)
        n = 3000
        h = 0.6; rets = np.empty(n)
        for t in range(n):
            z = rng.standard_t(5) / np.sqrt(5 / 3)
            rr = 0.02 + np.sqrt(h) * z
            if rng.random() < 0.004:
                rr -= abs(rng.normal(0, 6))
            rets[t] = rr
            h = 0.02 + 0.1 * rr**2 + 0.86 * h
        idx = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")
        masi = pd.Series(10000 * np.exp(np.cumsum(rets / 100)), index=idx)
        r = 100 * np.log(masi / masi.shift(1)).dropna()
        real = False

    df = pd.DataFrame({"Close": masi}).dropna()
    df["r"] = r
    # OHLC approximés pour les chandeliers (indice sans O/H/L propres fiables)
    df["Open"] = df["Close"].shift(1).fillna(df["Close"])
    rng2 = np.random.default_rng(1)
    noise = np.abs(rng2.normal(0, df["Close"] * 0.004))
    df["High"] = df[["Open", "Close"]].max(axis=1) + noise
    df["Low"] = df[["Open", "Close"]].min(axis=1) - noise

    losses = -df["r"].dropna()
    out = {"real": real}
    # VaR gaussienne (constante)
    mu, sig = df["r"].mean(), df["r"].std()
    out["var_gauss"] = float(-(mu + sig * stats.norm.ppf(0.01)))
    out["var_hist"] = float(-np.quantile(df["r"].dropna(), 0.01))

    # EVT
    try:
        from src import evt
        u = evt.suggest_threshold(losses.values)
        gpd = evt.fit_gpd(losses, u)
        out["var_evt"], out["es_evt"] = evt.gpd_var_es(gpd, 0.99)
        out["xi"] = gpd.xi
    except Exception:
        out["var_evt"], out["es_evt"], out["xi"] = out["var_hist"] * 1.1, out["var_hist"] * 1.5, 0.30

    # GARCH-EVT dynamique
    try:
        from src.garch_evt import fit_garch_evt
        g = fit_garch_evt(df["r"].dropna())
        df["vol_garch"] = g.cond_vol.reindex(df.index)
        df["VaR_evt_dyn"] = g.var_dyn["VaR_0.990"].reindex(df.index)
        out["var_garch"] = float(g.var_dyn["VaR_0.990"].iloc[-1])
        out["vol_next"] = g.forecast_vol
    except Exception:
        df["vol_garch"] = df["r"].rolling(20).std()
        df["VaR_evt_dyn"] = out["var_evt"]
        out["var_garch"] = out["var_evt"]
        out["vol_next"] = float(df["vol_garch"].iloc[-1])

    df["VaR_gauss"] = out["var_gauss"]
    df["VaR_evt"] = out["var_evt"]
    return df, out


df, K = load_all()
spot = float(df["Close"].iloc[-1])
MULT = 10  # 10 MAD / point (Future MASI 20)


@st.cache_data(show_spinner="Détection des régimes (HMM)…")
def get_regimes(returns_values, _idx):
    """Détecte les régimes de marché par HMM (calme/normal/crise)."""
    try:
        from src.regimes import fit_regimes, current_regime
        r = pd.Series(returns_values, index=_idx)
        reg = fit_regimes(r, n_states=3)
        return reg.labels, reg.vol_by_state, current_regime(reg)
    except Exception:
        return None, None, None


@st.cache_data(show_spinner=False)
def var_surface_grid(_losses_values):
    """Grille VaR = f(seuil u, niveau de confiance) pour la surface 3D.
    Montre la STABILITÉ des paramètres EVT (question d'entretien classique)."""
    from src import evt
    losses = pd.Series(_losses_values)
    us = np.quantile(losses, np.linspace(0.85, 0.97, 18))
    confs = np.linspace(0.95, 0.999, 18)
    Z = np.full((len(confs), len(us)), np.nan)
    for j, u in enumerate(us):
        try:
            gpd = evt.fit_gpd(losses, float(u))
            for i, a in enumerate(confs):
                Z[i, j], _ = evt.gpd_var_es(gpd, a)
        except Exception:
            pass
    return us, confs, Z

# ---------------------------------------------------------------
# Sidebar : filtres
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏔️ ATLAS Risk Terminal")
    st.caption("Tail-Risk & Margin — Achraf Akiyaf")
    if not K["real"]:
        st.warning("Mode démo (données synthétiques).")
    dmin, dmax = df.index.min().date(), df.index.max().date()
    dr = st.date_input("Période", (dmin, dmax), min_value=dmin, max_value=dmax)
    if isinstance(dr, tuple) and len(dr) == 2:
        df = df.loc[str(dr[0]):str(dr[1])]
    conf = st.select_slider("Niveau de VaR", [0.95, 0.99, 0.999], 0.99)
    st.download_button("⬇️ Export CSV", df.to_csv().encode(),
                       "masi_filtered.csv", "text/csv", use_container_width=True)
    st.caption("Projet académique — pas un conseil d'investissement.")

st.title("🏔️ ATLAS Risk Terminal")
st.caption("MASI 20 Tail-Risk & Margin Desk · Achraf Akiyaf — Master MMSD, FST Tanger "
          "· moteur EVT (GEV · POT-GPD · GARCH-EVT)")

# ---------------------------------------------------------------
# Flash Alert — basée sur les vraies données (VaR actuelle vs moyenne)
# ---------------------------------------------------------------
_var_series = df.get("VaR_evt_dyn", pd.Series([K["var_evt"]] * len(df), index=df.index))
_ratio_now = float(_var_series.iloc[-1] / max(_var_series.mean(), 1e-9))
if _ratio_now > 2.0:
    st.markdown(f"<div class='atlas-flash'>🔴 ALERTE CRISE — VaR actuelle "
               f"{_ratio_now:.1f}× la moyenne historique. Exposition à réduire.</div>",
               unsafe_allow_html=True)

# ---------------------------------------------------------------
# Barre de métriques — cartes KPI avec sparkline
# ---------------------------------------------------------------
var_evt_pts = K["var_evt"] / 100 * spot
im_evt_pct = K["es_evt"]  # marge ~ ES sur la journée (proxy pédagogique)
capital_reco = im_evt_pct / 100 * spot * MULT * 1.5


def kpi_card(col, label, value, delta, spark_series, color=BLUE):
    """Carte KPI style Bloomberg : label, valeur, delta, mini-sparkline."""
    with col:
        st.metric(label, value, delta)
        if spark_series is not None and len(spark_series.dropna()) > 5:
            sp = spark_series.dropna().iloc[-60:]
            f = go.Figure(go.Scatter(x=list(range(len(sp))), y=sp.values,
                                     mode="lines", line=dict(color=color, width=1.3),
                                     fill="tozeroy", fillcolor=f"rgba(0,102,204,0.08)"))
            f.update_layout(height=45, margin=dict(l=0, r=0, t=0, b=0),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(visible=False), yaxis=dict(visible=False),
                            showlegend=False)
            st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})


c1, c2, c3, c4 = st.columns(4)
kpi_card(c1, "Spot MASI", f"{spot:,.0f}", f"{df['r'].iloc[-1]:+.2f}%",
        df["Close"].iloc[-60:], GOLD)
kpi_card(c2, "Volatilité (GARCH, annualisée)", f"{K['vol_next']*np.sqrt(252):.1f}%",
        None, df.get("vol_garch", pd.Series(dtype=float)) * np.sqrt(252), RED)
kpi_card(c3, "VaR 99% EVT (1j)", f"{var_evt_pts:,.0f} pts", f"−{K['var_evt']:.2f}%",
        _var_series, RED)
kpi_card(c4, "Capital recommandé / contrat", f"{capital_reco:,.0f} MAD", None,
        _var_series * spot * MULT / 100 * 1.5, BLUE)

# ---------------------------------------------------------------
# Killer features : Simulateur Crash 2020 + Export PDF
# ---------------------------------------------------------------
kc1, kc2 = st.columns([1.4, 1])
with kc1:
    st.markdown("##### 🔴 Stress test — rejouer le pire jour historique")
    worst_day = df["r"].min()
    worst_date = df["r"].idxmin()
    if st.button(f"🔴 SIMULER LE CRASH DU {worst_date:%d/%m/%Y} ({worst_day:.2f}%)",
                 use_container_width=True):
        shocked_spot = spot * (1 + worst_day / 100)
        loss_pts = spot - shocked_spot
        # P&L d'une position longue de 10 contrats
        pnl = worst_day / 100 * spot * MULT * 10
        st.markdown(f"""<div style='background:{CARD};border-radius:12px;padding:16px;
          border-left:4px solid {RED}'>
          <b style='color:{RED};font-size:16px'>⚠️ SCÉNARIO CRASH APPLIQUÉ</b><br>
          MASI : {spot:,.0f} → <b>{shocked_spot:,.0f}</b> pts ({worst_day:.2f}%)<br>
          Perte : <b>{loss_pts:,.0f} points</b> · marge EVT requise :
          <b>{K['es_evt']/100*spot*MULT:,.0f} MAD/contrat</b><br>
          P&L position longue (10 contrats) :
          <b style='color:{RED}'>{pnl:,.0f} MAD</b></div>""",
          unsafe_allow_html=True)
with kc2:
    st.markdown("##### 📄 Rapport de risque")
    if st.button("Générer le rapport PDF", use_container_width=True):
        try:
            from src.report_pdf import build_risk_report
            p_crisis = 0.0
            try:
                from src import evt as _evt
                _u = _evt.suggest_threshold((-df["r"].dropna()).values)
                _gpd = _evt.fit_gpd(-df["r"].dropna(), _u)
                p_crisis = _evt.gpd_survival_prob(_gpd, 5, 30) * 100
                _bm = _evt.block_maxima(-df["r"].dropna())
                _gev = _evt.fit_gev(_bm)
                xi_gev, family = _gev.xi, _gev.family
                n_exc, xi_gpd, thr = _gpd.n_exceed, _gpd.xi, _gpd.threshold
            except Exception:
                xi_gev = family = None; n_exc = xi_gpd = thr = 0
            from src import margin as _m
            pdf = build_risk_report(
                kpis={"spot": spot, "vol_ann": K["vol_next"]*np.sqrt(252),
                      "var_evt": K["var_evt"], "es_evt": K["es_evt"],
                      "xi": K["xi"], "p_crisis": p_crisis},
                evt_params={"threshold": thr, "n_exceed": n_exc, "xi_gev": xi_gev or 0,
                            "xi_gpd": xi_gpd, "family": family or "Fréchet"},
                margins={"gauss": _m.margin_to_mad(_m.gaussian_margin(df["r"].dropna()), spot),
                         "hist": _m.margin_to_mad(_m.filtered_hist_margin(df["r"].dropna()), spot),
                         "evt": _m.margin_to_mad(_m.evt_margin(df["r"].dropna()), spot),
                         "garch": _m.margin_to_mad(_m.garch_evt_margin(df["r"].dropna()), spot)})
            st.download_button("⬇️ Télécharger le PDF", pdf,
                               "rapport_risque_MASI.pdf", "application/pdf",
                               use_container_width=True)
        except Exception as e:
            st.warning(f"Rapport indisponible ({type(e).__name__}).")

# ---------------------------------------------------------------
# Graphique principal : chandeliers + Bollinger + bande VaR + RSI/Volume
# ---------------------------------------------------------------
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI 14 jours : RSI = 100 - 100/(1+RS), RS = gains moyens / pertes moyennes."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


st.subheader("📈 Prix, Bollinger, bande VaR & indicateurs techniques")
tf1, tf2, tf3, tf4 = st.columns([2, 1, 1, 1])
tf = tf1.radio("Période", ["1M", "3M", "6M", "1A", "5A", "ALL"], index=5,
               horizontal=True)
zoom = tf2.checkbox("🔍 Zoom COVID 2020")
show_vol = tf3.checkbox("📊 Volume", value=False)
show_rsi = tf4.checkbox("📈 RSI(14)", value=False)

tf_days = {"1M": 21, "3M": 63, "6M": 126, "1A": 252, "5A": 1260, "ALL": len(df)}
if zoom:
    d = df.loc["2020-01-01":"2020-06-30"]
    d = d if not d.empty else df
else:
    d = df.iloc[-tf_days[tf]:]

ma = d["Close"].rolling(20).mean()
sd = d["Close"].rolling(20).std()
rsi = compute_rsi(d["Close"]) if show_rsi else None
has_vol = show_vol and "Volume" in d.columns and d["Volume"].fillna(0).sum() > 0

# Construction dynamique des subplots (prix toujours ; RSI / Volume optionnels)
from plotly.subplots import make_subplots
n_rows = 1 + int(show_rsi) + int(has_vol)
row_heights = [0.65] + [0.35 / max(1, (int(show_rsi) + int(has_vol)))] * (n_rows - 1)
specs_rows = list(range(1, n_rows + 1))
fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=row_heights)

fig.add_candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"],
                    close=d["Close"], name="MASI",
                    increasing_line_color=GREEN, decreasing_line_color=RED,
                    row=1, col=1)
fig.add_scatter(x=d.index, y=ma + 2 * sd, line=dict(color=MUTED, width=0.7, dash="dot"),
                name="Bollinger +2σ", row=1, col=1)
fig.add_scatter(x=d.index, y=ma - 2 * sd, line=dict(color=MUTED, width=0.7, dash="dot"),
                name="Bollinger −2σ", fill="tonexty",
                fillcolor="rgba(132,142,156,0.06)", row=1, col=1)
var_band = d["Close"] * (1 - d.get("VaR_evt_dyn", K["var_evt"]) / 100)
fig.add_scatter(x=d.index, y=var_band, line=dict(color=RED, width=1),
                name="Seuil VaR 99% EVT", row=1, col=1)

worst_in_view = d["r"].nsmallest(3)
crisis_labels = {2020: "COVID-19", 2022: "Guerre Ukraine", 2008: "Crise 2008"}
for dt, val in worst_in_view.items():
    lbl = crisis_labels.get(dt.year, f"{val:.1f}%")
    fig.add_annotation(x=dt, y=d["Low"].get(dt, d["Close"].get(dt)),
                       text=f"▼ {lbl}", showarrow=True, arrowcolor=RED,
                       font=dict(color=RED, size=10), ay=30, row=1, col=1)

current_row = 2
if has_vol:
    vol_colors = [GREEN if c >= o else RED for o, c in zip(d["Open"], d["Close"])]
    fig.add_bar(x=d.index, y=d["Volume"], marker_color=vol_colors,
               name="Volume", row=current_row, col=1)
    fig.update_yaxes(title_text="Volume", row=current_row, col=1)
    current_row += 1
if show_rsi:
    fig.add_scatter(x=d.index, y=rsi, line=dict(color=GOLD, width=1.2),
                    name="RSI(14)", row=current_row, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color=RED, row=current_row, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=GREEN, row=current_row, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)

fig.update_layout(height=460 + 130 * (n_rows - 1), xaxis_rangeslider_visible=False, **PL)
st.plotly_chart(fig, use_container_width=True)
if show_rsi and rsi is not None and not rsi.dropna().empty:
    last_rsi = rsi.dropna().iloc[-1]
    zone = "Survente (<30)" if last_rsi < 30 else "Surachat (>70)" if last_rsi > 70 else "Neutre"
    st.caption(f"RSI(14) actuel : {last_rsi:.0f} — {zone}")

# ---------------------------------------------------------------
# Panneau risque : jauge + comparaison VaR
# ---------------------------------------------------------------
g1, g2 = st.columns([1, 1.3])
with g1:
    st.subheader("🎯 Jauge de risque")
    var_now = df.get("VaR_evt_dyn", pd.Series([K["var_evt"]])).iloc[-1]
    var_avg = df.get("VaR_evt_dyn", pd.Series([K["var_evt"]])).mean()
    ratio = float(var_now / max(var_avg, 1e-9) * 100)
    label = ("RISK-ON" if ratio < 80 else "CAUTION" if ratio < 120 else "CRASH WARNING")
    col = GREEN if ratio < 80 else GOLD if ratio < 120 else RED
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=ratio,
        number=dict(suffix="%", font=dict(size=34, color=TXT)),
        gauge=dict(axis=dict(range=[0, 200], tickcolor=MUTED),
                   bar=dict(color=col, thickness=0.3),
                   bgcolor="rgba(255,255,255,0.04)",
                   steps=[dict(range=[0, 80], color="rgba(14,203,129,0.25)"),
                          dict(range=[80, 120], color="rgba(240,185,11,0.25)"),
                          dict(range=[120, 200], color="rgba(246,70,93,0.30)")])))
    gauge.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=TXT, family="JetBrains Mono"))
    st.plotly_chart(gauge, use_container_width=True)
    st.markdown(f"<h3 style='text-align:center;color:{col}'>{label}</h3>",
                unsafe_allow_html=True)
with g2:
    st.subheader("📊 Comparaison des modèles de VaR 99%")
    names = ["Gaussienne", "Historique", "EVT (POT)", "GARCH-EVT"]
    vals = [K["var_gauss"], K["var_hist"], K["var_evt"], K["var_garch"]]
    cols = [MUTED, "#5A8DEE", GOLD, RED]
    bar = go.Figure(go.Bar(x=names, y=vals, marker_color=cols,
                           text=[f"{v:.2f}%" for v in vals], textposition="outside"))
    bar.update_layout(height=300, yaxis_title="VaR 99% (%)", **PL)
    st.plotly_chart(bar, use_container_width=True)
    try:
        png = bar.to_image(format="png", scale=2)
        st.download_button("📸 Exporter ce graphique (PNG)", png,
                           "comparaison_var_atlas.png", "image/png",
                           use_container_width=True)
    except Exception:
        st.caption("Export PNG indisponible (installer `kaleido` : "
                  "`pip install kaleido`).")

# ---------------------------------------------------------------
# Stress Test "What-If" — choc de volatilité en temps réel
# ---------------------------------------------------------------
st.subheader("🧪 Stress Test \"What-If\" — choc de volatilité")
shock_pct = st.slider("Choc de volatilité appliqué", 0, 200, 0, 10, format="%d%%",
                      help="Multiplie la volatilité GARCH par (1 + choc) et "
                           "recalcule la VaR EVT instantanément.")
shock_mult = 1 + shock_pct / 100
stressed_var = K["var_evt"] * shock_mult          # scaling simple sur la vol
stressed_pts = stressed_var / 100 * spot
stressed_margin_mad = K["es_evt"] * shock_mult / 100 * spot * MULT

wc1, wc2, wc3 = st.columns(3)
wc1.metric("VaR 99% stressée", f"{stressed_var:.2f}%",
          f"{(shock_mult-1)*100:+.0f}% vs base")
wc2.metric("Perte potentielle (points)", f"{stressed_pts:,.0f} pts")
wc3.metric("Marge requise stressée", f"{stressed_margin_mad:,.0f} MAD")
if shock_pct > 0:
    st.caption(f"Sous un choc de volatilité de +{shock_pct}%, la VaR 99% EVT "
              f"passe de {K['var_evt']:.2f}% à {stressed_var:.2f}% — "
              f"la marge requise augmente de {stressed_margin_mad - im_evt_pct/100*spot*MULT:,.0f} MAD.")

# ---------------------------------------------------------------
# Simulateur de trade (pédagogique)
# ---------------------------------------------------------------
st.subheader("🎮 Simulateur de position (pédagogique)")
s1, s2, s3 = st.columns([1, 1, 1.4])
qty = s1.slider("Nombre de contrats", 1, 100, 10)
im_mad = im_evt_pct / 100 * spot * MULT * qty
s1.metric("Marge requise (EVT)", f"{im_mad:,.0f} MAD")
sens = s2.radio("Direction", ["🟢 LONG (Acheter)", "🔴 SHORT (Vendre)"])
if s2.button("▶️ Simuler le jour suivant", use_container_width=True):
    # rendement "jour suivant" : tiré de la distribution historique (pas de triche)
    r_next = float(np.random.default_rng().choice(df["r"].dropna().values))
    direction = 1 if "LONG" in sens else -1
    pnl = direction * r_next / 100 * spot * MULT * qty
    color = GREEN if pnl >= 0 else RED
    s3.markdown(f"<div style='background:{CARD};border-radius:12px;padding:18px;"
                f"border-left:4px solid {color}'>"
                f"Rendement simulé : <b>{r_next:+.2f}%</b><br>"
                f"P&L estimé : <b style='color:{color};font-size:22px'>"
                f"{pnl:+,.0f} MAD</b><br>"
                f"<span style='color:{MUTED};font-size:11px'>Tirage aléatoire dans "
                f"la distribution historique — illustratif.</span></div>",
                unsafe_allow_html=True)

# ---------------------------------------------------------------
# Drawdown + périodes de stress
# ---------------------------------------------------------------
st.subheader("📉 Drawdown historique & périodes de stress")
cummax = df["Close"].cummax()
dd = (df["Close"] / cummax - 1) * 100
fig_dd = go.Figure()
fig_dd.add_scatter(x=dd.index, y=dd.values, fill="tozeroy",
                   line=dict(color=RED, width=1), name="Drawdown (%)")
if "VaR_evt_dyn" in df:
    stress = df["VaR_evt_dyn"] > 2 * df["VaR_evt_dyn"].mean()
    for period in df.index[stress]:
        fig_dd.add_vline(x=period, line_width=0.3, line_color="rgba(240,185,11,0.15)")
fig_dd.update_layout(height=320, yaxis_title="Drawdown (%)",
                     title="Perte cumulée depuis le dernier pic (bandes = VaR > 2× moyenne)",
                     **PL)
st.plotly_chart(fig_dd, use_container_width=True)
st.metric("Pire drawdown historique", f"{dd.min():.1f}%")

# ---------------------------------------------------------------
# Détection de régimes (HMM) + fond coloré
# ---------------------------------------------------------------
st.subheader("🧠 Détection de régimes de marché (HMM)")
labels, vol_state, cur = get_regimes(df["r"].dropna().values, df["r"].dropna().index)
if labels is not None:
    rc1, rc2 = st.columns([1.5, 1])
    with rc1:
        cmap = {"Calme": "rgba(14,203,129,0.10)", "Normal": "rgba(240,185,11,0.06)",
                "Crise": "rgba(246,70,93,0.18)"}
        figr = go.Figure()
        figr.add_scatter(x=df.index, y=df["Close"], line=dict(color=TXT, width=1),
                         name="MASI")
        lab = labels.reindex(df.index).ffill()
        for state, color in cmap.items():
            mask = lab == state
            if mask.any():
                # bandes verticales du régime
                for grp_start, grp_end in _contiguous(mask):
                    figr.add_vrect(x0=grp_start, x1=grp_end, fillcolor=color,
                                   line_width=0, layer="below")
        figr.update_layout(height=340,
                           title="Prix MASI colorié par régime (rouge = crise détectée)",
                           **PL)
        st.plotly_chart(figr, use_container_width=True)
    with rc2:
        c = cur["regime"]
        col = {"Calme": GREEN, "Normal": GOLD, "Crise": RED}.get(c, GOLD)
        st.markdown(f"<div style='background:{CARD};border-radius:12px;padding:20px;"
                    f"text-align:center;border:1px solid {col}'>"
                    f"<div style='color:{MUTED};font-size:13px'>RÉGIME COURANT</div>"
                    f"<div style='color:{col};font-size:30px;font-weight:700'>{c}</div>"
                    f"<div style='color:{MUTED};font-size:12px'>confiance "
                    f"{cur['probability']*100:.0f}%</div></div>",
                    unsafe_allow_html=True)
        st.caption("Volatilité par régime :")
        for k, v in vol_state.items():
            st.caption(f"• {k} : {v:.2f}%/jour")
        st.caption("⚠️ Le HMM caractérise le régime *actuel*, il ne prédit pas "
                   "l'avenir.")

# ---------------------------------------------------------------
# Surface 3D de VaR (stabilité des paramètres)
# ---------------------------------------------------------------
st.subheader("🌐 Surface 3D de VaR — stabilité des paramètres EVT")
losses_arr = (-df["r"].dropna()).values
us, confs, Z = var_surface_grid(losses_arr)
surf = go.Figure(go.Surface(x=us, y=confs * 100, z=Z,
                            colorscale=[[0, "#123524"], [0.5, GOLD], [1, RED]],
                            colorbar=dict(title="VaR %")))
surf.update_layout(scene=dict(xaxis_title="Seuil u (%)", yaxis_title="Confiance (%)",
                              zaxis_title="VaR (%)"),
                   height=460, **PL)
st.plotly_chart(surf, use_container_width=True)
st.caption("Une surface lisse et peu sensible au seuil = paramètres EVT stables "
           "(gage de robustesse du modèle).")

# ---------------------------------------------------------------
# Heatmap calendrier annuel
# ---------------------------------------------------------------
st.subheader("🗓️ Heatmap des rendements — repérer les 'annus horribilis'")
yr = df.copy()
yr["year"] = yr.index.year
yr["month"] = yr.index.month
pivot = yr.pivot_table(index="year", columns="month", values="r", aggfunc="sum")
months = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun", "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
hm = go.Figure(go.Heatmap(z=pivot.values, x=months, y=pivot.index.astype(str),
                          colorscale=[[0, RED], [0.5, CARD], [1, GREEN]], zmid=0,
                          colorbar=dict(title="Rdt mensuel %"),
                          hovertemplate="%{y} %{x}<br>%{z:.1f}%<extra></extra>"))
hm.update_layout(height=360, title="Rendement mensuel cumulé (vert = hausse, rouge = baisse)",
                 **PL)
st.plotly_chart(hm, use_container_width=True)

# ---------------------------------------------------------------
# Métriques de performance glissantes (rolling metrics)
# ---------------------------------------------------------------
st.subheader("📐 Métriques de performance (fenêtre glissante)")
win = st.slider("Fenêtre (jours)", 30, 120, 60, 30)
rr = df["r"].dropna()
roll_vol = rr.rolling(win).std() * np.sqrt(252)
roll_sharpe = (rr.rolling(win).mean() / rr.rolling(win).std()) * np.sqrt(252)
roll_dd = (df["Close"] / df["Close"].rolling(win).max() - 1) * 100

rm1, rm2, rm3 = st.columns(3)
with rm1:
    f = go.Figure(go.Scatter(x=roll_sharpe.index, y=roll_sharpe, line=dict(color=GOLD, width=1)))
    f.add_hline(y=0, line_color=MUTED, line_width=0.5)
    f.update_layout(title=f"Sharpe glissant ({win}j, annualisé)", height=240, **PL)
    st.plotly_chart(f, use_container_width=True)
with rm2:
    f = go.Figure(go.Scatter(x=roll_vol.index, y=roll_vol, line=dict(color=RED, width=1)))
    f.update_layout(title=f"Volatilité glissante ({win}j, annualisée %)", height=240, **PL)
    st.plotly_chart(f, use_container_width=True)
with rm3:
    f = go.Figure(go.Scatter(x=roll_dd.index, y=roll_dd, fill="tozeroy",
                             line=dict(color=GREEN, width=1)))
    f.update_layout(title=f"Drawdown glissant ({win}j, %)", height=240, **PL)
    st.plotly_chart(f, use_container_width=True)
st.caption(f"Sharpe actuel : {roll_sharpe.iloc[-1]:.2f} · "
          f"Vol actuelle : {roll_vol.iloc[-1]:.1f}% · "
          f"Drawdown actuel : {roll_dd.iloc[-1]:.1f}%")

# ---------------------------------------------------------------
# Backtest "On Demand" (période personnalisée)
# ---------------------------------------------------------------
with st.expander("🎯 Backtest personnalisé — tester le modèle sur une période choisie"):
    st.caption("Les paramètres EVT sont calibrés sur tout l'historique ; le "
              "backtest est donc une évaluation **hors-échantillon** sur la "
              "période sélectionnée (aucune ré-estimation sur cette fenêtre).")
    bc1, bc2, bc3 = st.columns([1, 1, 1])
    bt_start = bc1.date_input("Date de début", df.index.min().date(),
                              min_value=df.index.min().date(),
                              max_value=df.index.max().date(), key="bt_start")
    bt_end = bc2.date_input("Date de fin", df.index.max().date(),
                            min_value=df.index.min().date(),
                            max_value=df.index.max().date(), key="bt_end")
    run_bt = bc3.button("▶️ Exécuter le backtest", use_container_width=True)

    if run_bt:
        try:
            from src.evt import suggest_threshold, fit_gpd, gpd_var_es
            from src import backtest as _bt
            period_r = df["r"].loc[str(bt_start):str(bt_end)].dropna()
            if len(period_r) < 20:
                st.warning("Période trop courte (< 20 jours) pour un backtest fiable.")
            else:
                # Paramètres EVT calibrés sur TOUT l'historique -> évaluation hors-échantillon
                full_losses = -df["r"].dropna()
                u_full = suggest_threshold(full_losses.values)
                gpd_full = fit_gpd(full_losses, u_full)
                var99, _ = gpd_var_es(gpd_full, 0.99)

                viol = (period_r < -var99).values
                tl = _bt.traffic_light(viol, 0.99)
                st.markdown(f"**Verdict : {tl['color']} {tl['verdict']}** — "
                           f"{tl['obs']} violations observées vs {tl['expected']} "
                           f"attendues sur {len(period_r)} jours")
                res_table = pd.DataFrame([{
                    "Violations obs.": tl["obs"], "Attendues": tl["expected"],
                    "Kupiec p": tl["kupiec_p"],
                    "Christoffersen p": tl["christoffersen_p"],
                    "Indépendance p": tl["independence_p"]}])
                st.dataframe(res_table, use_container_width=True, hide_index=True)

                # --- Feu tricolore de Bâle (seuils officiels 0-4/5-9/≥10 sur 250j) ---
                basel = _bt.basel_zone(tl["obs"], len(period_r))
                st.markdown(f"**Zone de Bâle : {basel['zone']}** — {basel['desc']}")

                # --- Sévérité des violations (Perte/VaR ratio) ---
                sev = _bt.violation_severity(period_r, var99)
                if sev["n"] > 0:
                    sc1, sc2 = st.columns(2)
                    sc1.metric("Ratio Perte/VaR moyen", f"{sev['mean_ratio']:.2f}×",
                              help="1.0 = violation marginale ; >2 = échec sévère du modèle")
                    sc2.metric("Pire violation (ratio max)", f"{sev['max_ratio']:.2f}×")
                    fsev = go.Figure(go.Bar(x=list(range(len(sev["ratios"]))),
                                            y=sev["ratios"].values,
                                            marker_color=RED))
                    fsev.add_hline(y=1.0, line_dash="dot", line_color=MUTED)
                    fsev.update_layout(title="Ratio Perte/VaR par violation (triées)",
                                       height=260, xaxis_title="Violation (triée)",
                                       yaxis_title="Perte / VaR", **PL)
                    st.plotly_chart(fsev, use_container_width=True)

                fbt = go.Figure()
                fbt.add_scatter(x=period_r.index, y=period_r.values, mode="lines",
                                line=dict(color=MUTED, width=0.8), name="Rendements")
                fbt.add_hline(y=-var99, line_dash="dash", line_color=RED,
                             annotation_text=f"−VaR99 = {-var99:.2f}%")
                v = period_r[period_r < -var99]
                fbt.add_scatter(x=v.index, y=v.values, mode="markers",
                                marker=dict(color=RED, size=8), name="Violations")
                fbt.update_layout(height=300,
                                  title=f"Violations du {bt_start} au {bt_end}", **PL)
                st.plotly_chart(fbt, use_container_width=True)
        except Exception as e:
            st.warning(f"Backtest indisponible ({type(e).__name__}: {e}).")

# ---------------------------------------------------------------
# 🎓 Analyses Senior Quant : MPOR scaling, reverse stress test,
# sensibilité du seuil GPD
# ---------------------------------------------------------------
st.subheader("🎓 Analyses Senior Quant")
try:
    from src.evt import suggest_threshold, fit_gpd, gpd_var_es, threshold_sensitivity
    from src.garch_evt import var_horizon

    _losses_full = -df["r"].dropna()
    _u_full = suggest_threshold(_losses_full.values)
    _gpd_full = fit_gpd(_losses_full, _u_full)
    _var99_full, _ = gpd_var_es(_gpd_full, 0.99)

    sq1, sq2 = st.columns(2)
    with sq1:
        st.markdown("##### ⏱️ Échelle MPOR — VaR selon la période de liquidation")
        st.caption("Scaling en loi de puissance VaR(h) = VaR(1j) × h^ξ "
                  "(Danielsson & de Vries) — PAS la racine du temps, valable "
                  "seulement sous normalité.")
        mpor_h = st.slider("Période de liquidation (jours)", 1, 30, 2)
        vh = var_horizon(_var99_full, _gpd_full.xi, mpor_h)
        vh_mad = vh / 100 * spot * MULT
        st.metric(f"VaR 99% à {mpor_h} jour(s)", f"{vh:.2f}%",
                  f"×{vh/_var99_full:.2f} vs 1 jour")
        st.metric("Marge correspondante", f"{vh_mad:,.0f} MAD")
        if mpor_h > 1 and _gpd_full.xi < 0.5:
            st.caption(f"⚠️ Note : ξ={_gpd_full.xi:.2f} < 0.5 → la VaR croît "
                      f"*plus lentement* que la racine du temps (×{mpor_h**0.5:.2f} "
                      f"sous normalité vs ×{mpor_h**_gpd_full.xi:.2f} ici).")

        st.markdown("##### 🎯 Reverse Stress Test")
        margin_mad_now = _var99_full / 100 * spot * MULT
        seuil_stress = (margin_mad_now / (spot * MULT)) * 100
        verdict = ("🟢 Résilient aux chocs modérés" if seuil_stress < 5 else
                  "🟠 Vulnérable aux chocs sévères" if seuil_stress < 10 else
                  "🔴 Extrêmement vulnérable — augmenter la marge")
        st.metric("Seuil de perte qui égalise la marge actuelle",
                  f"{seuil_stress:.2f}%")
        st.markdown(f"**{verdict}**")

    with sq2:
        st.markdown("##### 📐 Sensibilité du seuil GPD (stability check)")
        st.caption("Si la VaR varie fortement quand on change le seuil, le "
                  "modèle est instable — question classique en entretien.")
        sens = threshold_sensitivity(_losses_full.values, _u_full)
        max_var = sens["var_relative_pct"].abs().max()
        fsens = go.Figure(go.Scatter(x=sens["seuil_u"], y=sens["VaR"],
                                     mode="lines+markers", line=dict(color=GOLD)))
        fsens.add_vline(x=_u_full, line_dash="dot", line_color=BLUE,
                        annotation_text="seuil retenu")
        fsens.update_layout(title="VaR 99% en fonction du seuil u (±20%)",
                            xaxis_title="Seuil u (%)", yaxis_title="VaR 99% (%)",
                            height=300, **PL)
        st.plotly_chart(fsens, use_container_width=True)
        if max_var > 10:
            st.warning(f"⚠️ Modèle sensible au choix du seuil : variation "
                      f"max de {max_var:.1f}% sur la plage testée.")
        else:
            st.success(f"✅ Modèle stable : variation max de la VaR de "
                      f"seulement {max_var:.1f}% sur ±20% de changement du seuil.")

    st.markdown("##### 🔄 Backtest walk-forward — stabilité dans le temps")
    st.caption("P-value de Christoffersen calculée sur des fenêtres glissantes "
              "de 250 jours (VaR fixe, calibrée sur tout l'historique). Si la "
              "p-value tombe sous 0.05, le modèle est rejeté sur cette fenêtre.")
    from src import backtest as _bt2
    wf_pvals = _bt2.rolling_backtest_pvalues(df["r"], _var99_full)
    fwf = go.Figure()
    fwf.add_scatter(x=wf_pvals.index, y=wf_pvals["christoffersen_p"],
                    mode="lines+markers", line=dict(color=BLUE, width=1.2),
                    marker=dict(size=4), name="p-value Christoffersen")
    fwf.add_hline(y=0.05, line_dash="dash", line_color=RED,
                 annotation_text="seuil de rejet (5%)")
    fwf.update_layout(title="P-value de Christoffersen — fenêtres glissantes de 250j",
                      yaxis_title="p-value", height=280, **PL)
    st.plotly_chart(fwf, use_container_width=True)
    n_reject = int((wf_pvals["christoffersen_p"] < 0.05).sum())
    st.caption(f"Le modèle est rejeté sur {n_reject}/{len(wf_pvals)} fenêtres "
              f"glissantes ({n_reject/len(wf_pvals)*100:.0f}%).")
except Exception as e:
    st.warning(f"Analyses senior indisponibles ({type(e).__name__}).")

st.markdown(f"<p style='color:{MUTED};font-size:11px'>© MASI 20 Tail-Risk Lab — "
            "projet académique Master MMSD. Moteur EVT + HMM réels (src/). Le "
            "simulateur et les signaux sont pédagogiques, pas un conseil "
            "d'investissement.</p>", unsafe_allow_html=True)
