"""Modèles classiques de référence (la « ligne de base » à battre)."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def classical_models(seed: int = 0) -> dict:
    """Trois références de complexité croissante."""
    return {
        "Régression logistique": LogisticRegression(max_iter=1000),
        "SVM (noyau RBF)": SVC(kernel="rbf", gamma="scale"),
        "Petit réseau de neurones": MLPClassifier(
            hidden_layer_sizes=(16, 16), max_iter=2000, random_state=seed
        ),
    }
