"""
MASI Tail-Risk Lab — GARCH-EVT conditionnel (McNeil & Frey, 2000)
==================================================================
GJR-GARCH(1,1) pour capturer les grappes de volatilité et l'effet de
levier, puis POT-GPD sur les résidus standardisés -> VaR/CVaR dynamiques.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evt import GPDResult, fit_gpd, gpd_var_es


@dataclass
class GarchEvtResult:
    cond_vol: pd.Series       # volatilité conditionnelle sigma_t
    std_resid: pd.Series      # résidus standardisés z_t
    gpd: GPDResult            # GPD ajustée sur les pertes standardisées
    var_dyn: pd.DataFrame     # VaR/ES dynamiques par niveau
    forecast_vol: float       # sigma_{t+1}
    forecast_mean: float
    params: dict


def fit_garch_evt(returns: pd.Series,
                  levels: tuple[float, ...] = (0.95, 0.99, 0.999),
                  q_threshold: float = 0.90) -> GarchEvtResult:
    """Pipeline GJR-GARCH(1,1)-t -> POT sur résidus -> VaR dynamique."""
    from arch import arch_model

    am = arch_model(returns, mean="Constant", vol="GARCH",
                    p=1, o=1, q=1, dist="t")     # o=1 => terme GJR (levier)
    fit = am.fit(disp="off")
    mu = float(fit.params.get("mu", 0.0))
    sigma_t = fit.conditional_volatility
    z = (returns - mu) / sigma_t

    losses_z = pd.Series(-z.values, index=z.index)
    u = float(np.quantile(losses_z, q_threshold))
    gpd = fit_gpd(losses_z, u)

    fc = fit.forecast(horizon=1, reindex=False)
    sig_next = float(np.sqrt(fc.variance.values[-1, 0]))

    frames = {}
    for a in levels:
        qz, esz = gpd_var_es(gpd, a)
        frames[f"VaR_{a:.3f}"] = -(mu - sigma_t * qz)
        frames[f"ES_{a:.3f}"] = -(mu - sigma_t * esz)
    var_dyn = pd.DataFrame(frames, index=returns.index)

    return GarchEvtResult(cond_vol=sigma_t, std_resid=z, gpd=gpd,
                          var_dyn=var_dyn, forecast_vol=sig_next,
                          forecast_mean=mu, params=dict(fit.params))


def var_horizon(var_1d: float, xi: float, days: int) -> float:
    """Projection de la VaR à h jours (règle de scaling en loi de puissance).

    Pour les queues lourdes (Danielsson & de Vries), le scaling correct
    n'est PAS la racine du temps (h^0.5, valable seulement sous normalité)
    mais VaR(h) = VaR(1) * h^xi lorsque xi > 0. Note contre-intuitive :
    pour xi < 0.5 (cas fréquent en pratique), cette croissance est PLUS
    LENTE que la racine du temps — la queue lourde domine à horizon court,
    mais l'agrégation temporelle "dilue" moins vite que sous normalité.
    """
    expo = xi if xi > 0 else 0.5
    return float(var_1d * days ** expo)
