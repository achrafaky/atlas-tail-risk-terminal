"""
MASI Tail-Risk Lab — Détection de régimes de marché (HMM)
==========================================================
Auteur : Achraf Akiyaf — Master MMSD, FST Tanger.

Modèle de Markov caché (Gaussian HMM) sur les rendements pour identifier
des régimes latents de volatilité : calme / normal / crise.

⚠️ HONNÊTETÉ SCIENTIFIQUE : un HMM identifie le régime le PLUS PROBABLE À
L'INSTANT t (filtrage/décodage de Viterbi). Il ne PRÉDIT PAS l'avenir —
il caractérise l'état courant du marché. C'est un outil de diagnostic de
régime, pas une boule de cristal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegimeResult:
    states: pd.Series          # état décodé (0/1/2) par date
    labels: pd.Series          # étiquette lisible ("Calme"/"Normal"/"Crise")
    vol_by_state: dict         # volatilité moyenne de chaque état
    proba: pd.DataFrame        # probabilités a posteriori de chaque régime
    crisis_state: int          # index de l'état "crise" (vol la plus haute)


def fit_regimes(returns: pd.Series, n_states: int = 3,
                seed: int = 42) -> RegimeResult:
    """Ajuste un Gaussian HMM à `n_states` régimes sur les rendements.

    Les états sont réordonnés par volatilité croissante pour une
    interprétation stable (0 = plus calme, n-1 = crise).
    """
    from hmmlearn.hmm import GaussianHMM

    X = returns.values.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="diag",
                        n_iter=200, random_state=seed)
    model.fit(X)
    raw_states = model.predict(X)
    proba = model.predict_proba(X)

    # Réordonner les états par volatilité (variance) croissante
    variances = model.covars_.flatten()
    order = np.argsort(variances)                 # calme -> crise
    remap = {old: new for new, old in enumerate(order)}
    states = np.array([remap[s] for s in raw_states])
    proba = proba[:, order]

    labels_map = ({0: "Calme", 1: "Normal", 2: "Crise"} if n_states == 3
                  else {i: f"Régime {i}" for i in range(n_states)})
    vol_by_state = {labels_map[i]: float(np.sqrt(variances[order[i]]))
                    for i in range(n_states)}

    return RegimeResult(
        states=pd.Series(states, index=returns.index, name="state"),
        labels=pd.Series([labels_map[s] for s in states], index=returns.index,
                         name="regime"),
        vol_by_state=vol_by_state,
        proba=pd.DataFrame(proba, index=returns.index,
                           columns=[labels_map[i] for i in range(n_states)]),
        crisis_state=n_states - 1,
    )


def current_regime(res: RegimeResult) -> dict:
    """Régime courant (dernière date) + sa probabilité."""
    label = res.labels.iloc[-1]
    p = float(res.proba.iloc[-1].max())
    return {"regime": label, "probability": p}
