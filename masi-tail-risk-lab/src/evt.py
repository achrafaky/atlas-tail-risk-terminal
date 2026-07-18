"""
MASI Tail-Risk Lab — Moteur EVT (Statistique des Valeurs Extrêmes)
===================================================================
Auteur : Achraf Akiyaf — Master MMSD, FST Tanger.

Implémente les deux approches fondatrices de l'EVT + la version conditionnelle :

1. Bloc-Maxima (GEV)  — théorème de Fisher-Tippett-Gnedenko.
2. Peaks Over Threshold (GPD) — théorème de Pickands-Balkema-de Haan,
   avec choix du seuil par Mean Residual Life et Parameter Stability,
   VaR/ES en formules fermées (McNeil & Frey), IC bootstrap paramétrique.
3. Hill estimator — indice de queue non paramétrique.

Convention : on modélise les PERTES (losses = -rendements). Un "maximum"
correspond donc à une pire chute. ATTENTION scipy : le paramètre `c` de
genextreme/genpareto vaut -xi dans la convention standard d'EVT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import genextreme, genpareto


# =====================================================================
# 1. APPROCHE BLOC-MAXIMA — LOI GEV
# =====================================================================
@dataclass
class GEVResult:
    xi: float           # indice de queue (xi > 0 : Fréchet, queue lourde)
    mu: float           # localisation
    sigma: float        # échelle
    n_blocks: int
    family: str
    loglik: float
    lr_pvalue_gumbel: float   # test du rapport de vraisemblance H0 : xi = 0
    maxima: np.ndarray = field(repr=False, default=None)


def block_maxima(losses: pd.Series, freq: str = "ME") -> np.ndarray:
    """Maxima par bloc (mensuel par défaut) de la série des pertes."""
    bm = losses.resample(freq).max().dropna()
    bm = bm[bm > 0]
    return bm.values


def fit_gev(maxima: np.ndarray) -> GEVResult:
    """Ajuste une loi GEV par maximum de vraisemblance + test de Gumbel.

    Le test du rapport de vraisemblance compare GEV (xi libre) à Gumbel
    (xi = 0) : une p-value faible rejette Gumbel au profit d'une vraie
    queue lourde (Fréchet).
    """
    c, mu, sigma = genextreme.fit(maxima)
    xi = -c
    ll_gev = genextreme.logpdf(maxima, c=c, loc=mu, scale=sigma).sum()

    loc_g, scale_g = stats.gumbel_r.fit(maxima)
    ll_gum = stats.gumbel_r.logpdf(maxima, loc=loc_g, scale=scale_g).sum()
    lr = 2 * (ll_gev - ll_gum)
    p_gumbel = float(1 - stats.chi2.cdf(max(lr, 0), df=1))

    family = ("Fréchet (queue lourde)" if xi > 0.05
              else "Weibull (queue bornée)" if xi < -0.05
              else "Gumbel (queue légère)")
    return GEVResult(xi=float(xi), mu=float(mu), sigma=float(sigma),
                     n_blocks=len(maxima), family=family,
                     loglik=float(ll_gev), lr_pvalue_gumbel=p_gumbel,
                     maxima=maxima)


def gev_return_level(res: GEVResult, T_blocks: np.ndarray) -> np.ndarray:
    """Niveau de retour à T blocs : la perte dépassée en moyenne 1 fois / T mois."""
    T = np.asarray(T_blocks, dtype=float)
    y = -np.log(1 - 1.0 / T)
    if abs(res.xi) < 1e-6:
        return res.mu - res.sigma * np.log(y)
    return res.mu + res.sigma / res.xi * (y ** (-res.xi) - 1)


# =====================================================================
# 2. APPROCHE POT — LOI GPD
# =====================================================================
@dataclass
class GPDResult:
    xi: float
    beta: float          # échelle
    threshold: float     # seuil u
    n_exceed: int
    n_total: int
    loglik: float
    exceedances: np.ndarray = field(repr=False, default=None)


def mean_residual_life(losses: np.ndarray, n_points: int = 60) -> pd.DataFrame:
    """Mean Residual Life Plot : e(u) = E[X - u | X > u].

    Au-delà d'un bon seuil, e(u) devient approximativement LINÉAIRE en u
    (propriété caractéristique de la GPD). On lit le seuil au début de
    cette zone linéaire.
    """
    qs = np.linspace(0.80, 0.995, n_points)
    rows = []
    for q in qs:
        u = np.quantile(losses, q)
        exc = losses[losses > u] - u
        if len(exc) >= 10:
            se = exc.std(ddof=1) / np.sqrt(len(exc))
            rows.append((u, q, exc.mean(),
                         exc.mean() - 1.96 * se, exc.mean() + 1.96 * se, len(exc)))
    return pd.DataFrame(rows, columns=["u", "q", "mrl", "lo", "hi", "n"])


def parameter_stability(losses: np.ndarray, n_points: int = 40) -> pd.DataFrame:
    """Parameter Stability Plot : xi estimé en fonction du seuil u.

    Au-delà d'un bon seuil, xi(u) se STABILISE. On choisit u dans cette zone.
    """
    qs = np.linspace(0.85, 0.985, n_points)
    rows = []
    for q in qs:
        u = np.quantile(losses, q)
        exc = losses[losses > u] - u
        if len(exc) >= 20:
            try:
                xi, _, beta = genpareto.fit(exc, floc=0)
                rows.append((u, q, xi, beta, len(exc)))
            except Exception:
                pass
    return pd.DataFrame(rows, columns=["u", "q", "xi", "beta", "n"])


def suggest_threshold(losses: np.ndarray) -> float:
    """Seuil automatique : zone de stabilité de xi (variance minimale sur
    fenêtre glissante), avec garde-fou n_exceed >= 100. Sert de point de
    départ objectif — à confirmer visuellement par les deux plots."""
    ps = parameter_stability(losses)
    if len(ps) < 8:
        return float(np.quantile(losses, 0.90))
    xi = ps["xi"].values
    w = 6
    var = [xi[i:i + w].var() for i in range(len(xi) - w)]
    u = float(ps["u"].iloc[int(np.argmin(var))])
    if (losses > u).sum() < 100:
        u = float(np.quantile(losses, 0.90))
    return u


def fit_gpd(losses: pd.Series | np.ndarray, threshold: float) -> GPDResult:
    """Ajuste une GPD par MLE sur les excès au-dessus du seuil."""
    vals = losses.values if isinstance(losses, pd.Series) else np.asarray(losses)
    exc = vals[vals > threshold] - threshold
    xi, _, beta = genpareto.fit(exc, floc=0)
    ll = genpareto.logpdf(exc, c=xi, loc=0, scale=beta).sum()
    return GPDResult(xi=float(xi), beta=float(beta), threshold=float(threshold),
                     n_exceed=len(exc), n_total=len(vals),
                     loglik=float(ll), exceedances=exc)


def gpd_var_es(res: GPDResult, alpha: float) -> tuple[float, float]:
    """VaR et Expected Shortfall (CVaR) au niveau alpha — formules fermées
    de McNeil & Frey. Retourne des pertes positives (en %)."""
    zeta = res.n_exceed / res.n_total
    xi, beta, u = res.xi, res.beta, res.threshold
    if abs(xi) < 1e-8:
        var = u + beta * np.log(zeta / (1 - alpha))
    else:
        var = u + beta / xi * (((1 - alpha) / zeta) ** (-xi) - 1)
    es = (var + beta - xi * u) / (1 - xi) if xi < 1 else np.inf
    return float(var), float(es)


def gpd_survival_prob(res: GPDResult, loss_level: float, horizon_days: int = 30) -> float:
    """P(au moins un jour avec perte > loss_level sur horizon_days jours)."""
    zeta = res.n_exceed / res.n_total
    if loss_level <= res.threshold:
        p_day = zeta
    else:
        p_day = zeta * float(genpareto.sf(loss_level - res.threshold,
                                          c=res.xi, scale=res.beta))
    return float(1 - (1 - p_day) ** horizon_days)


def gpd_return_level(res: GPDResult, T_days: np.ndarray) -> np.ndarray:
    """Niveau de retour à T jours (perte dépassée en moyenne 1 fois / T jours)."""
    zeta = res.n_exceed / res.n_total
    T = np.asarray(T_days, dtype=float)
    if abs(res.xi) < 1e-8:
        return res.threshold + res.beta * np.log(T * zeta)
    return res.threshold + res.beta / res.xi * ((T * zeta) ** res.xi - 1)


def gpd_return_level_ci(res: GPDResult, T_days: np.ndarray,
                        n_boot: int = 1000, seed: int = 0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """IC 95 % par bootstrap paramétrique (rééchantillonnage sous la GPD ajustée)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(T_days)))
    for b in range(n_boot):
        sample = genpareto.rvs(c=res.xi, scale=res.beta,
                               size=res.n_exceed, random_state=rng)
        try:
            xi, _, beta = genpareto.fit(sample, floc=0)
            tmp = GPDResult(xi=xi, beta=beta, threshold=res.threshold,
                            n_exceed=res.n_exceed, n_total=res.n_total, loglik=0)
            out[b] = gpd_return_level(tmp, T_days)
        except Exception:
            out[b] = np.nan
    return (np.nanpercentile(out, 2.5, axis=0),
            np.nanpercentile(out, 97.5, axis=0))


# =====================================================================
# 3. HILL ESTIMATOR
# =====================================================================
def hill_estimator(losses: np.ndarray, k_max: int | None = None) -> pd.DataFrame:
    """Indice de queue de Hill xi(k) pour k = 10..k_max (alpha = 1/xi)."""
    x = np.sort(losses[losses > 0])[::-1]
    n = len(x)
    k_max = k_max or min(n - 1, 500)
    logs = np.log(x)
    csum = np.cumsum(logs)
    ks = np.arange(10, k_max)
    xis = np.array([csum[k - 1] / k - logs[k] for k in ks])
    se = xis / np.sqrt(ks)
    return pd.DataFrame({"k": ks, "xi": xis, "alpha": 1 / np.maximum(xis, 1e-9),
                         "lo": xis - 1.96 * se, "hi": xis + 1.96 * se})


def threshold_sensitivity(losses: np.ndarray, base_threshold: float,
                          alpha: float = 0.99, pct_range: float = 0.20,
                          n_points: int = 9) -> pd.DataFrame:
    """Sensibilité du modèle GPD au choix du seuil u (stability check).

    Fait varier u de -pct_range à +pct_range autour du seuil de base et
    recalcule (xi, beta, VaR). Si la VaR varie de plus de ~10% sur cette
    plage, le modèle est jugé sensible au seuil (avertissement à afficher).
    """
    factors = np.linspace(1 - pct_range, 1 + pct_range, n_points)
    rows = []
    base_var = None
    for f in factors:
        u = base_threshold * f
        try:
            res = fit_gpd(pd.Series(losses), u)
            var, _ = gpd_var_es(res, alpha)
            rows.append({"seuil_u": u, "xi": res.xi, "beta": res.beta,
                        "n_exceed": res.n_exceed, "VaR": var})
            if abs(f - 1.0) < 1e-9:
                base_var = var
        except Exception:
            rows.append({"seuil_u": u, "xi": np.nan, "beta": np.nan,
                        "n_exceed": np.nan, "VaR": np.nan})
    df = pd.DataFrame(rows)
    if base_var:
        df["var_relative_pct"] = (df["VaR"] / base_var - 1) * 100
    return df
