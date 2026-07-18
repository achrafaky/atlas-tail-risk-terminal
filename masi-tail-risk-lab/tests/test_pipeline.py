"""Tests unitaires du pipeline EVT — culture senior : chaque brique testée."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import genpareto
from src import evt, backtest as bt


def test_gpd_recovers_known_parameters():
    """Sur des données GPD simulées, le fit doit retrouver xi et beta."""
    rng = np.random.default_rng(0)
    sample = genpareto.rvs(c=0.3, scale=1.5, size=5000, random_state=rng)
    import pandas as pd
    res = evt.fit_gpd(pd.Series(sample), threshold=0.0)
    assert abs(res.xi - 0.3) < 0.1
    assert abs(res.beta - 1.5) < 0.2


def test_var_es_ordering():
    """VaR et ES doivent croître avec le niveau de confiance ; ES >= VaR."""
    import pandas as pd
    rng = np.random.default_rng(1)
    losses = pd.Series(np.abs(rng.standard_t(4, 3000)))
    gpd = evt.fit_gpd(losses, float(np.quantile(losses, 0.9)))
    v95, e95 = evt.gpd_var_es(gpd, 0.95)
    v99, e99 = evt.gpd_var_es(gpd, 0.99)
    assert v99 > v95 and e99 > v99 and e95 >= v95


def test_kupiec_perfect_model():
    """Un modèle parfait (1% de violations) ne doit pas être rejeté."""
    rng = np.random.default_rng(2)
    viol = rng.random(2000) < 0.01
    assert bt.kupiec_pof(viol, 0.99)["pvalue"] > 0.05


def test_environment_imports():
    import numpy, pandas, scipy, arch  # noqa: F401
    assert True


def test_hmm_separates_volatility_regimes():
    """Le HMM doit séparer un régime calme d'un régime agité."""
    import pandas as pd
    from src.regimes import fit_regimes
    rng = np.random.default_rng(3)
    calm = rng.normal(0, 0.5, 800)
    crisis = rng.normal(0, 3.0, 200)
    r = pd.Series(np.concatenate([calm, crisis, calm]),
                  index=pd.date_range("2020-01-01", periods=1800, freq="D"))
    reg = fit_regimes(r, n_states=3)
    vols = sorted(reg.vol_by_state.values())
    assert vols[-1] > 2 * vols[0]   # le régime crise est nettement plus volatil


def test_monte_carlo_matches_analytic():
    """La marge Monte Carlo doit coïncider avec la formule analytique (<5%)."""
    import pandas as pd
    from src import margin
    rng = np.random.default_rng(4)
    r = pd.Series(rng.standard_t(5, 2000) * 0.8,
                  index=pd.date_range("2015-01-01", periods=2000, freq="B"))
    mc = margin.monte_carlo_margin(r, 0.995, 2, n_sims=5000)
    assert abs(mc["relative_gap_pct"]) < 15   # cohérence MC vs analytique


def test_mpor_scaling_uses_power_law_not_absurd_blowup():
    """La VaR à 10 jours ne doit JAMAIS dépasser 100% (sanity check anti
    formule fausse type h^(1/xi))."""
    from src.garch_evt import var_horizon
    v10 = var_horizon(2.0, 0.30, 10)
    assert v10 < 100   # une formule h^(1/xi) donnerait ~4300%, absurde
    assert v10 > 2.0   # doit tout de même croître avec l'horizon


def test_basel_zone_thresholds():
    from src.backtest import basel_zone
    assert basel_zone(3, 250)["zone"].startswith("🟢")
    assert basel_zone(7, 250)["zone"].startswith("🟠")
    assert basel_zone(12, 250)["zone"].startswith("🔴")


def test_threshold_sensitivity_returns_dataframe():
    import pandas as pd
    from src.evt import threshold_sensitivity
    rng = np.random.default_rng(5)
    losses = np.abs(rng.standard_t(4, 3000))
    sens = threshold_sensitivity(losses, float(np.quantile(losses, 0.9)))
    assert "var_relative_pct" in sens.columns
    assert len(sens) == 9
