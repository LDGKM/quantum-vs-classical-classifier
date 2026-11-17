"""Comparaison équitable des modèles : mêmes données, mêmes métriques."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .classical import classical_models
from .data import load_dataset
from .quantum import QuantumKernelSVC, VariationalQuantumClassifier


def _n_params(model) -> int | None:
    if hasattr(model, "coef_"):
        return int(model.coef_.size + model.intercept_.size)
    if hasattr(model, "coefs_"):
        return int(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_))
    if isinstance(model, VariationalQuantumClassifier):
        return model.n_params_
    return None  # SVM : modèle non paramétrique (dépend des vecteurs de support)


def run_benchmark(
    dataset: str = "moons",
    n_qubits: int = 2,
    n_layers: int = 2,
    steps: int = 60,
    n_samples: int = 200,
    seed: int = 42,
    include_kernel: bool = True,
    verbose: bool = True,
    out_dir: str | Path | None = "results",
) -> pd.DataFrame:
    """Entraîne tous les modèles et renvoie un tableau de résultats."""
    X_train, X_test, y_train, y_test = load_dataset(
        dataset, n_features=n_qubits, n_samples=n_samples, seed=seed
    )

    models = dict(classical_models(seed))
    models["VQC (quantique)"] = VariationalQuantumClassifier(
        n_layers=n_layers, steps=steps, seed=seed
    )
    if include_kernel:
        models["SVM à noyau quantique"] = QuantumKernelSVC()

    rows = []
    for name, model in models.items():
        if verbose:
            print(f"Entraînement : {name} ...")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        y_pred = model.predict(X_test)
        rows.append(
            {
                "modèle": name,
                "famille": "quantique" if "quantique" in name else "classique",
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "temps_entraînement_s": train_time,
                "nb_paramètres": _n_params(model),
            }
        )

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    df["nb_paramètres"] = df["nb_paramètres"].astype("Int64")  # entiers, avec valeurs manquantes
    df.attrs.update(dataset=dataset, n_qubits=n_qubits, n_layers=n_layers, steps=steps, seed=seed)

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{dataset}_{n_qubits}q"
        df.to_csv(out / f"{stem}.csv", index=False)
        # Tableau LaTeX prêt à être inclus dans docs/rapport.tex
        df.to_latex(
            out / f"{stem}.tex",
            index=False,
            float_format="%.3f",
            columns=["modèle", "accuracy", "f1", "temps_entraînement_s", "nb_paramètres"],
            header=["Modèle", "Accuracy", "F1", "Temps (s)", "Paramètres"],
            na_rep="--",
        )
        _plot(df, out / f"{stem}.png")
    return df


def _plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#6a4c93" if f == "quantique" else "#8d99ae" for f in df["famille"]]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(df["modèle"], df["accuracy"], color=colors)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy sur le jeu de test")
    ax.set_title(f"{df.attrs.get('dataset')} — {df.attrs.get('n_qubits')} qubits")
    for i, v in enumerate(df["accuracy"]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
