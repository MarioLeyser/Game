"""Datasets de los 4 mundos de Quantum Defense.

Clase +1 = Llamitas de Coherencia, clase -1 = Slimes de Entropía.
Todos los datos se escalan a [0, pi] (rango natural para ZZFeatureMap).
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

WORLD_META = {
    1: {"id": "valle-lineal", "name": "El Valle Lineal", "model": "svm-linear"},
    2: {"id": "meseta-radial", "name": "La Meseta Radial", "model": "svm-poly-rbf"},
    3: {"id": "cumbres-vortice", "name": "Las Cumbres del Vórtice", "model": "qsvc-zz"},
    4: {"id": "abismo-decoherencia", "name": "El Abismo de Decoherencia", "model": "qsvc-zz-noisy"},
}


def _scale(X: np.ndarray) -> np.ndarray:
    lo, hi = X.min(axis=0), X.max(axis=0)
    return (X - lo) / (hi - lo) * np.pi


def _split(X, y, n_train, rng):
    idx = rng.permutation(len(X))
    tr, te = idx[:n_train], idx[n_train:]
    return X[tr], y[tr], X[te], y[te]


N_INFILTRATORS = {1: 0, 2: 1, 3: 1, 4: 3}


def build_world(world: int, seed: int = 7, n_train: int = 30, n_test: int = 60):
    """Devuelve (X_train, y_train, X_test, y_test, outlier_mask_train)."""
    rng = np.random.default_rng(seed + world)
    n = n_train + n_test
    if world == 1:
        X, y = make_blobs(n_samples=n, centers=[(-1.5, -1.0), (1.5, 1.0)],
                          cluster_std=0.8, random_state=seed)
    elif world == 2:
        X, y = make_circles(n_samples=n, factor=0.4, noise=0.07, random_state=seed)
    elif world == 3:
        X, y = make_moons(n_samples=n, noise=0.12, random_state=seed)
    elif world == 4:
        X, y = make_moons(n_samples=n, noise=0.22, random_state=seed)
    else:
        raise ValueError("world debe estar en 1..4")

    X = _scale(X)
    y = np.where(y == 1, 1, -1)
    Xtr, ytr, Xte, yte = _split(X, y, n_train, rng)

    outliers = np.zeros(len(Xtr), dtype=bool)
    k = N_INFILTRATORS[world]
    if k:
        # Slimes infiltrados: etiquetas invertidas (ruido). Sobreajustarlos rompe el escudo.
        pick = rng.choice(len(Xtr), size=k, replace=False)
        ytr = ytr.copy()
        ytr[pick] *= -1
        outliers[pick] = True
    return Xtr, ytr, Xte, yte, outliers
