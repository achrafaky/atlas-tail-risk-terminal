"""
MASI Tail-Risk Lab — Modèles de marge initiale du Future MASI 20
=================================================================
Auteur : Achraf Akiyaf — Master MMSD, FST Tanger.

Une chambre de compensation (CCP) calcule la marge initiale (IM) comme la
perte potentielle sur la période de liquidation (MPOR, 2 jours) à un niveau
de confiance élevé. On compare trois modèles et on les convertit en dirhams :

    IM (MAD) = VaR_{MPOR, alpha} (en %) / 100 x Niveau_indice x multiplicateur

Multiplicateur du Future MASI 20 : 10 MAD par point d'indice.
Dépôt de garantie initial de référence communiqué : 1000 MAD.

Modèles :
  1. gaussian_margin       — z_alpha * sigma_MPOR
  2. filtered_hist_margin  — quantile empirique MPOR sur fenêtre glissante
  3. evt_margin            — VaR MPOR par GPD (POT)

⚠️ Analyse pédagogique : la marge réelle de la CCP Maroc dépend de
paramètres officiels non publics. On raisonne en ordre de grandeur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .evt import fit_gpd, gpd_var_es


def mpor_returns(returns: pd.Series, mpor: int = 2) -> pd.Series:
    """Rendements agrégés sur la période de liquidation (somme de log-rendements)."""
    return returns.rolling(mpor).sum().dropna()


def gaussian_margin(returns: pd.Series, alpha: float = 0.995,
                    mpor: int = 2) -> float:
    """Marge gaussienne : VaR MPOR sous hypothèse normale (en % de l'indice)."""
    r_h = mpor_returns(returns, mpor)
    mu, sigma = r_h.mean(), r_h.std()
    return float(-(mu + sigma * stats.norm.ppf(1 - alpha)))


def filtered_hist_margin(returns: pd.Series, alpha: float = 0.995,
                         mpor: int = 2) -> float:
    """Marge historique : quantile empirique des pertes MPOR (en %)."""
    r_h = mpor_returns(returns, mpor)
    return float(-np.quantile(r_h, 1 - alpha))


def evt_margin(returns: pd.Series, alpha: float = 0.995, mpor: int = 2,
               q_threshold: float = 0.90) -> float:
    """Marge EVT : VaR MPOR par POT-GPD sur les pertes agrégées (en %)."""
    losses_h = -mpor_returns(returns, mpor)
    u = float(np.quantile(losses_h, q_threshold))
    gpd = fit_gpd(losses_h, u)
    var, _ = gpd_var_es(gpd, alpha)
    return float(var)


def margin_to_mad(margin_pct: float, index_level: float,
                  multiplier: float = 10.0) -> float:
    """Convertit une marge en % en dirhams pour un contrat."""
    return margin_pct / 100.0 * index_level * multiplier


def margin_timeseries(returns: pd.Series, prices: pd.Series,
                      model: str = "evt", alpha: float = 0.995,
                      mpor: int = 2, window: int = 500,
                      multiplier: float = 10.0) -> pd.DataFrame:
    """Série temporelle de la marge (en % et en MAD) sur fenêtre glissante.

    À chaque date t, la marge n'utilise que les `window` observations
    passées (walk-forward, pas de look-ahead).
    """
    idx = returns.index[window:]
    pct = np.empty(len(idx))
    for i, t in enumerate(idx):
        sub = returns.iloc[i:i + window]
        if model == "gaussian":
            pct[i] = gaussian_margin(sub, alpha, mpor)
        elif model == "hist":
            pct[i] = filtered_hist_margin(sub, alpha, mpor)
        else:  # evt
            try:
                pct[i] = evt_margin(sub, alpha, mpor)
            except Exception:
                pct[i] = filtered_hist_margin(sub, alpha, mpor)
    lvl = prices.reindex(idx).values
    mad = pct / 100.0 * lvl * multiplier
    return pd.DataFrame({"margin_pct": pct, "index_level": lvl,
                         "margin_mad": mad}, index=idx)


def procyclicality(margin_mad: pd.Series) -> dict:
    """Mesure de procyclicité : ratio pic/plancher et hausse max sur 1 mois."""
    peak, trough = float(margin_mad.max()), float(margin_mad.min())
    monthly_rise = margin_mad.pct_change(21).max()
    return {"peak_mad": peak, "trough_mad": trough,
            "peak_to_trough_ratio": peak / max(trough, 1e-9),
            "max_monthly_rise_pct": float(monthly_rise * 100)}


# =====================================================================
# MARGE CONDITIONNELLE GARCH-EVT (la marge "réglementaire" moderne)
# =====================================================================
def garch_evt_margin(returns: pd.Series, alpha: float = 0.995,
                     mpor: int = 2) -> float:
    """Marge conditionnelle GARCH-EVT (en %).

    Contrairement à la marge EVT inconditionnelle (procyclique), celle-ci
    tient compte de la volatilité ACTUELLE : VaR_t = mu + sigma_{t+1} *
    q_z(alpha), où q_z est le quantile EVT des résidus standardisés. La
    projection sur la période de liquidation MPOR suit la racine du temps.

    C'est l'approche recommandée par les régulateurs (marges conditionnelles)
    car elle évite les à-coups violents de la marge inconditionnelle.
    """
    from .garch_evt import fit_garch_evt
    from .evt import gpd_var_es
    g = fit_garch_evt(returns)
    qz, _ = gpd_var_es(g.gpd, alpha)             # quantile EVT des résidus
    var_1d = -(g.forecast_mean - g.forecast_vol * qz)   # VaR 1j conditionnelle
    return float(var_1d * np.sqrt(mpor))         # scaling MPOR (racine du temps)


# =====================================================================
# STRESS TESTS PAR SCÉNARIOS (le quotidien du risk manager)
# =====================================================================
def stress_margin(returns: pd.Series, prices: pd.Series,
                  scenario: str = "vol_shock", alpha: float = 0.995,
                  mpor: int = 2, multiplier: float = 10.0) -> dict:
    """Recalcule la marge sous un scénario de stress.

    Scénarios :
      - 'vol_shock'   : volatilité multipliée par 2 (+100 %)
      - 'flash_crash' : choc de -10 % sur 2 jours ajouté à la queue
      - 'covid_2020'  : recalibrage sur la seule fenêtre de crise mars 2020
    Retourne la marge stressée (% et MAD) et le ratio vs marge normale.
    """
    base_pct = evt_margin(returns, alpha, mpor)
    spot = float(prices.iloc[-1])

    if scenario == "vol_shock":
        stressed = returns * 2.0                 # double la dispersion
        s_pct = evt_margin(stressed, alpha, mpor)
        desc = "Choc de volatilité +100 %"
    elif scenario == "flash_crash":
        shock = pd.Series([-5.0, -5.0], index=returns.index[-2:])  # -10% sur 2j
        stressed = pd.concat([returns, shock])
        s_pct = evt_margin(stressed, alpha, mpor)
        desc = "Flash crash -10 % sur 2 jours"
    elif scenario == "covid_2020":
        window = returns.loc["2020-02-01":"2020-05-31"]
        s_pct = evt_margin(window, alpha, mpor) if len(window) > 50 else base_pct * 2.5
        desc = "Recalibrage sur la crise COVID (fév-mai 2020)"
    else:
        s_pct = base_pct
        desc = "Aucun stress"

    return {
        "scenario": desc,
        "base_margin_pct": base_pct,
        "stressed_margin_pct": s_pct,
        "base_margin_mad": margin_to_mad(base_pct, spot, multiplier),
        "stressed_margin_mad": margin_to_mad(s_pct, spot, multiplier),
        "increase_pct": (s_pct / base_pct - 1) * 100,
    }


# =====================================================================
# MARGE PAR SIMULATION MONTE CARLO (méthode des CCP)
# =====================================================================
def monte_carlo_margin(returns: pd.Series, alpha: float = 0.995, mpor: int = 2,
                       n_sims: int = 10000, q_threshold: float = 0.90,
                       seed: int = 0) -> dict:
    """Marge par simulation Monte Carlo (comme les vraies CCP).

    On simule n_sims trajectoires de pertes sur MPOR jours en tirant dans un
    modèle semi-paramétrique : corps de la distribution ré-échantillonné
    (bootstrap historique), queue au-delà du seuil tirée dans la GPD ajustée.
    La marge = quantile alpha de la distribution simulée des pertes MPOR.
    On la compare à la marge analytique (formule fermée) pour validation.
    """
    from scipy.stats import genpareto
    from .evt import fit_gpd, gpd_var_es

    rng = np.random.default_rng(seed)
    losses = -returns.dropna()
    u = float(np.quantile(losses, q_threshold))
    gpd = fit_gpd(losses, u)
    zeta = gpd.n_exceed / gpd.n_total
    body = losses[losses <= u].values

    def sample_daily(size):
        out = np.empty(size)
        is_tail = rng.random(size) < zeta
        n_tail = int(is_tail.sum())
        out[~is_tail] = rng.choice(body, size - n_tail)
        out[is_tail] = u + genpareto.rvs(c=gpd.xi, scale=gpd.beta,
                                         size=n_tail, random_state=rng)
        return out

    mpor_losses = sum(sample_daily(n_sims) for _ in range(mpor))
    mc_margin = float(np.quantile(mpor_losses, alpha))
    analytic, _ = gpd_var_es(gpd, alpha)
    analytic_mpor = analytic * np.sqrt(mpor)

    return {
        "mc_margin_pct": mc_margin,
        "analytic_margin_pct": float(analytic_mpor),
        "n_sims": n_sims,
        "relative_gap_pct": (mc_margin / analytic_mpor - 1) * 100,
        "simulated_losses": mpor_losses,
    }
