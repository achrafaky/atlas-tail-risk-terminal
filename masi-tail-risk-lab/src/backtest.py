"""
MASI Tail-Risk Lab — Backtesting réglementaire
===============================================
Tests de Kupiec (proportion de violations) et Christoffersen (indépendance
et couverture conditionnelle), avec verdict feu tricolore façon Bâle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def kupiec_pof(violations: np.ndarray, alpha: float) -> dict:
    """H0 : la fréquence de violations vaut 1 - alpha (couverture correcte)."""
    v = np.asarray(violations, dtype=bool)
    n, x = len(v), int(v.sum())
    p = 1 - alpha
    pi = x / n if n else 0.0
    if x in (0, n):
        lr = abs(-2 * (n * np.log(1 - p) if x == 0 else n * np.log(p)))
    else:
        lr = -2 * (x * np.log(p / pi) + (n - x) * np.log((1 - p) / (1 - pi)))
    return {"n": n, "violations": x, "expected": n * p,
            "LR": float(lr), "pvalue": float(1 - stats.chi2.cdf(lr, df=1))}


def christoffersen_independence(violations: np.ndarray) -> dict:
    """H0 : les violations sont indépendantes (pas de clustering)."""
    v = np.asarray(violations, dtype=int)
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(v[:-1], v[1:]):
        n00 += (a == 0) & (b == 0); n01 += (a == 0) & (b == 1)
        n10 += (a == 1) & (b == 0); n11 += (a == 1) & (b == 1)
    pi01 = n01 / max(1, n00 + n01)
    pi11 = n11 / max(1, n10 + n11)
    pi = (n01 + n11) / max(1, n00 + n01 + n10 + n11)

    def ll(p, a, b):
        return 0.0 if p in (0, 1) or a + b == 0 else a * np.log(1 - p) + b * np.log(p)

    lr = max(0.0, -2 * (ll(pi, n00 + n10, n01 + n11)
                        - (ll(pi01, n00, n01) + ll(pi11, n10, n11))))
    return {"LR": float(lr), "pvalue": float(1 - stats.chi2.cdf(lr, df=1))}


def christoffersen_cc(violations: np.ndarray, alpha: float) -> dict:
    """Couverture conditionnelle : LR_cc = LR_pof + LR_ind ~ chi2(2)."""
    pof = kupiec_pof(violations, alpha)
    ind = christoffersen_independence(violations)
    lr = pof["LR"] + ind["LR"]
    return {"LR": float(lr), "pvalue": float(1 - stats.chi2.cdf(lr, df=2))}


def traffic_light(violations: np.ndarray, alpha: float) -> dict:
    """Verdict complet + couleur (vert/orange/rouge)."""
    pof = kupiec_pof(violations, alpha)
    ind = christoffersen_independence(violations)
    cc = christoffersen_cc(violations, alpha)
    obs, exp = pof["violations"], pof["expected"]
    if cc["pvalue"] >= 0.05 and pof["pvalue"] >= 0.05:
        color, verdict = "🟢", "Validé"
    elif cc["pvalue"] >= 0.01 or obs <= 1.5 * max(exp, 1):
        color, verdict = "🟠", "Sous-estimation légère"
    else:
        color, verdict = "🔴", "Rejeté"
    return {"color": color, "verdict": verdict, "obs": obs,
            "expected": round(exp, 1),
            "kupiec_p": round(pof["pvalue"], 4),
            "christoffersen_p": round(cc["pvalue"], 4),
            "independence_p": round(ind["pvalue"], 4)}


def basel_zone(n_violations: int, n_obs: int = 250) -> dict:
    """Zone du comité de Bâle (backtesting VaR 99%, fenêtre de 250 jours).

    Seuils officiels (Bâle II, Amendment to the Capital Accord) :
      Verte  : 0-4 violations  -> modèle validé, facteur multiplicateur 3.0
      Jaune  : 5-9 violations  -> zone de surveillance, facteur croissant 3.4-3.85
      Rouge  : >=10 violations -> modèle rejeté, facteur multiplicateur 4.0
    Les seuils sont définis pour une fenêtre de 250 jours ; on les met à
    l'échelle proportionnellement si n_obs diffère.
    """
    scaled = n_violations * 250 / max(n_obs, 1)
    if scaled < 5:
        return {"zone": "🟢 Verte", "color": "#0ECB81",
                "desc": "Modèle validé (facteur multiplicateur 3.0)"}
    elif scaled < 10:
        return {"zone": "🟠 Jaune", "color": "#F0B90B",
                "desc": "Zone de surveillance (facteur croissant 3.4-3.85)"}
    else:
        return {"zone": "🔴 Rouge", "color": "#F6465D",
                "desc": "Modèle rejeté (facteur multiplicateur 4.0)"}


def violation_severity(returns, var_level: float) -> dict:
    """Analyse de sévérité : ratio Perte/VaR pour chaque jour de violation.

    Un ratio proche de 1 = violation marginale. Un ratio élevé = le modèle
    a échoué gravement ce jour-là (queue non capturée).
    """
    import numpy as np
    losses = -returns
    viol_losses = losses[losses > var_level]
    if len(viol_losses) == 0:
        return {"n": 0, "mean_ratio": None, "max_ratio": None, "ratios": None}
    ratios = (viol_losses / var_level).sort_values(ascending=False)
    return {"n": len(ratios), "mean_ratio": float(ratios.mean()),
            "max_ratio": float(ratios.max()), "ratios": ratios}


def rolling_backtest_pvalues(returns, var_level: float, window: int = 250,
                             step: int = 21) -> pd.DataFrame:
    """Walk-forward léger : p-value de Christoffersen sur des fenêtres
    glissantes de `window` jours (par défaut 250 = 1 an), tous les `step`
    jours. La VaR est FIXE (calibrée en amont) ; seule la fenêtre
    d'observation glisse — permet de voir si le modèle reste stable dans
    le temps sans recalibrer à chaque pas (coût de calcul raisonnable).
    """
    import numpy as np
    losses = -returns.dropna()
    idx = losses.index
    rows = []
    for start in range(0, len(losses) - window, step):
        w = losses.iloc[start:start + window]
        viol = (w > var_level).values
        pof = kupiec_pof(viol, 0.99)
        ind = christoffersen_independence(viol)
        cc = christoffersen_cc(viol, 0.99)
        rows.append({"date": w.index[-1], "n_viol": int(viol.sum()),
                    "kupiec_p": pof["pvalue"], "christoffersen_p": cc["pvalue"]})
    return pd.DataFrame(rows).set_index("date")
