"""
MASI Tail-Risk Lab — Utilitaires de données
============================================
Auteur : Achraf Akiyaf — Master MMSD, FST Tanger.

Chargement, audit qualité et transformation de l'historique de l'indice
MASI (Bourse de Casablanca), export Investing.com au format anglais
(dates mm/dd/yyyy, nombres "17,987.82").
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_masi(path: str = "data/raw/masi.csv") -> pd.Series:
    """Charge le CSV Investing.com et renvoie la série des prix de clôture.

    Le fichier a les colonnes : Date, Price, Open, High, Low, Vol., Change %.
    On ne conserve que la clôture (`Price`).

    Returns
    -------
    pd.Series : prix de clôture, index = dates trié croissant.
    """
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["Close"] = (df["Price"].astype(str)
                   .str.replace(",", "", regex=False)   # "17,987.82" -> "17987.82"
                   .astype(float))
    return df.set_index("Date")["Close"].sort_index()


def audit_and_clean(prices: pd.Series) -> tuple[pd.Series, list[str]]:
    """Audit qualité + nettoyage documenté (traçabilité risk management).

    Contrôles : doublons de dates, valeurs manquantes, prix <= 0,
    variations quotidiennes > 20 % (probables erreurs de données).
    Correction : suppression des doublons, interpolation temporelle
    SUR LES PRIX (jamais sur les rendements, pour ne pas biaiser les queues).

    Returns
    -------
    (série nettoyée, journal des corrections)
    """
    log: list[str] = []
    s = prices.copy()

    ndup = int(s.index.duplicated().sum())
    if ndup:
        s = s[~s.index.duplicated(keep="last")]
        log.append(f"{ndup} doublon(s) de dates supprimé(s).")

    nbad = int((s <= 0).sum())
    if nbad:
        s[s <= 0] = np.nan
        log.append(f"{nbad} prix <= 0 mis à NaN.")

    jumps = s.pct_change().abs()
    njump = int((jumps > 0.20).sum())
    if njump:
        log.append(f"{njump} variation(s) > 20 %/jour détectée(s) (conservées, "
                   "possibles vrais chocs — à inspecter au cas par cas).")

    nnan = int(s.isna().sum())
    if nnan:
        s = s.interpolate(method="time", limit_direction="both")
        log.append(f"{nnan} valeur(s) manquante(s) interpolée(s) (sur les prix).")

    if not log:
        log.append("Aucune anomalie détectée — données propres.")
    return s.dropna(), log


def log_returns(prices: pd.Series) -> pd.Series:
    """Log-rendements quotidiens en % : r_t = 100 * ln(P_t / P_{t-1})."""
    return 100.0 * np.log(prices / prices.shift(1)).dropna()


def stylized_facts(returns: pd.Series) -> dict:
    """Faits stylisés des rendements (pour le diagnostic de non-normalité)."""
    from scipy import stats
    jb = stats.jarque_bera(returns)
    return {
        "n": len(returns),
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "skewness": float(stats.skew(returns)),
        "excess_kurtosis": float(stats.kurtosis(returns)),
        "min": float(returns.min()),
        "max": float(returns.max()),
        "jarque_bera_stat": float(jb.statistic),
        "jarque_bera_pvalue": float(jb.pvalue),
        "worst_5_days": returns.nsmallest(5),
    }
