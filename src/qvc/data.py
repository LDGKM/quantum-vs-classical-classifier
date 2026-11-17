"""Chargement et préparation des jeux de données.

Un circuit quantique a peu de qubits : on réduit donc les données à `n_features`
dimensions (une par qubit) avec une ACP, puis on les met à l'échelle dans [0, pi]
pour pouvoir les encoder comme des angles de rotation.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, make_circles, make_moons
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATASETS = ("moons", "circles", "iris", "breast_cancer")


def _raw(name: str, n_samples: int, noise: float, seed: int):
    if name == "moons":
        return make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    if name == "circles":
        return make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
    if name == "iris":
        X, y = load_iris(return_X_y=True)
        mask = y < 2  # versicolor vs setosa : problème binaire
        return X[mask], y[mask]
    if name == "breast_cancer":
        return load_breast_cancer(return_X_y=True)
    raise ValueError(f"Jeu de données inconnu : {name!r}. Choix : {DATASETS}")


def load_dataset(
    name: str = "moons",
    n_features: int = 2,
    n_samples: int = 200,
    noise: float = 0.15,
    test_size: float = 0.3,
    seed: int = 42,
):
    """Renvoie X_train, X_test, y_train, y_test (y dans {0, 1}, X dans [0, pi]).

    La réduction (ACP) et la mise à l'échelle sont apprises sur le train uniquement,
    pour ne pas laisser fuiter d'information du test.
    """
    X, y = _raw(name, n_samples, noise, seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    steps = [StandardScaler()]
    if X.shape[1] > n_features:
        steps.append(PCA(n_components=n_features, random_state=seed))
    elif X.shape[1] < n_features:
        raise ValueError(
            f"{name!r} n'a que {X.shape[1]} variables : n_features doit être <= {X.shape[1]}"
        )
    steps.append(MinMaxScaler(feature_range=(0, np.pi)))
    prep = make_pipeline(*steps)

    X_train = prep.fit_transform(X_train)
    # Les points de test peuvent dépasser légèrement [0, pi] : on les borne.
    X_test = np.clip(prep.transform(X_test), 0, np.pi)
    return X_train, X_test, y_train, y_test
