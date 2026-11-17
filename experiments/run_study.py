"""Étude expérimentale complète (résultats de l'article docs/article.tex).

Trois expériences, chacune répétée sur plusieurs graines aléatoires :
  A. tous les modèles sur 4 jeux de données (5 graines) ;
  B. montée en charge : Breast Cancer de 2 à 6 qubits (3 graines) ;
  C. expressivité : VQC sur « circles » avec 1 à 4 couches (5 graines).

    python experiments/run_study.py            # ~10 min sur 2 cœurs
    python experiments/run_study.py --quick    # version réduite pour tester

Sorties dans results/study/ : raw.csv, tableaux LaTeX (.tex) et figures (.pdf/.png).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

from qvc import QuantumKernelSVC, VariationalQuantumClassifier, classical_models, load_dataset  # noqa: E402

OUT = ROOT / "results" / "study"
STEPS = 60
DATASETS = {"moons": 2, "circles": 2, "iris": 4, "breast_cancer": 4}
NICE = {"moons": "Moons", "circles": "Circles", "iris": "Iris", "breast_cancer": "Breast Cancer"}
SHORT = {
    "Régression logistique": "LogReg",
    "SVM (noyau RBF)": "SVM-RBF",
    "Petit réseau de neurones": "MLP",
    "VQC (quantique)": "VQC",
    "SVM à noyau quantique": "QSVM",
}


def n_params(model):
    if hasattr(model, "coef_"):
        return model.coef_.size + model.intercept_.size
    if hasattr(model, "coefs_"):
        return sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
    return getattr(model, "n_params_", np.nan)


def run_job(job: dict) -> list[dict]:
    """Un jeu de données, un nombre de qubits, une graine : entraîne les modèles demandés."""
    X_tr, X_te, y_tr, y_te = load_dataset(job["dataset"], n_features=job["qubits"], seed=job["seed"])
    models = {}
    if job.get("classical", True):
        models.update(classical_models(job["seed"]))
    models["VQC (quantique)"] = VariationalQuantumClassifier(
        n_layers=job.get("layers", 2), steps=STEPS, seed=job["seed"]
    )
    if job.get("kernel", True):
        models["SVM à noyau quantique"] = QuantumKernelSVC()

    rows = []
    for name, m in models.items():
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        dt = time.perf_counter() - t0
        rows.append({
            "experiment": job["experiment"], "dataset": job["dataset"], "qubits": job["qubits"],
            "layers": job.get("layers", 2), "seed": job["seed"], "model": SHORT[name],
            "family": "quantum" if name in ("VQC (quantique)", "SVM à noyau quantique") else "classical",
            "train_acc": accuracy_score(y_tr, m.predict(X_tr)),
            "test_acc": accuracy_score(y_te, m.predict(X_te)),
            "time_s": dt, "params": n_params(m),
        })
    return rows


def build_jobs(quick: bool) -> list[dict]:
    seeds5 = range(2 if quick else 5)
    seeds3 = range(1 if quick else 3)
    jobs = [dict(experiment="A", dataset=d, qubits=q, seed=s) for d, q in DATASETS.items() for s in seeds5]
    jobs += [dict(experiment="B", dataset="breast_cancer", qubits=q, seed=s)
             for q in ((2, 4) if quick else range(2, 7)) for s in seeds3]
    jobs += [dict(experiment="C", dataset="circles", qubits=2, layers=L, seed=s, classical=False, kernel=False)
             for L in ((1, 3) if quick else range(1, 5)) for s in seeds5]
    # les plus longs d'abord, pour mieux répartir sur les cœurs
    return sorted(jobs, key=lambda j: -j["qubits"])


# ------------------------------------------------------------------------------ sorties
def table_A(df: pd.DataFrame) -> str:
    a = df[df.experiment == "A"]
    g = a.groupby(["model", "dataset"])["test_acc"].agg(["mean", "std"]).reset_index()
    order = ["LogReg", "SVM-RBF", "MLP", "VQC", "QSVM"]
    best = g.groupby("dataset")["mean"].max()
    lines = [r"\begin{tabular}{l" + "c" * len(DATASETS) + "}", r"\toprule",
             "Modèle & " + " & ".join(NICE[d] for d in DATASETS) + r" \\", r"\midrule"]
    for m in order:
        cells = []
        for d in DATASETS:
            r = g[(g.model == m) & (g.dataset == d)].iloc[0]
            txt = f"{r['mean']:.3f} $\\pm$ {r['std']:.3f}"
            cells.append(r"\textbf{" + txt + "}" if np.isclose(r["mean"], best[d]) else txt)
        if m == "VQC":
            lines.append(r"\midrule")
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_cost(df: pd.DataFrame) -> str:
    a = df[(df.experiment == "A") & (df.dataset == "breast_cancer")]
    g = a.groupby("model").agg(time=("time_s", "mean"), params=("params", "first")).reindex(
        ["LogReg", "SVM-RBF", "MLP", "VQC", "QSVM"])
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Modèle & Temps (s) & Paramètres \\", r"\midrule"]
    for m, r in g.iterrows():
        p = "--" if pd.isna(r["params"]) else f"{int(r['params'])}"
        lines.append(f"{m} & {r['time']:.3f} & {p} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def figures(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"LogReg": "#c9ced6", "SVM-RBF": "#8d99ae", "MLP": "#4a5568", "VQC": "#9b7fc1", "QSVM": "#6a4c93"}

    # A : accuracy par jeu de données
    a = df[df.experiment == "A"]
    g = a.groupby(["dataset", "model"])["test_acc"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    w = 0.16
    for i, m in enumerate(colors):
        means = [g.loc[(d, m), "mean"] for d in DATASETS]
        stds = [g.loc[(d, m), "std"] for d in DATASETS]
        ax.bar(np.arange(len(DATASETS)) + (i - 2) * w, means, w, yerr=stds, label=m,
               color=colors[m], capsize=1.5, error_kw={"lw": 0.6})
    ax.set_xticks(range(len(DATASETS)), [NICE[d] for d in DATASETS])
    ax.set_ylim(0.5, 1.02); ax.set_ylabel("Accuracy (test)")
    ax.legend(ncol=5, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_datasets.pdf"); fig.savefig(OUT / "fig_datasets.png", dpi=200)
    plt.close(fig)

    # B : montée en qubits
    b = df[df.experiment == "B"]
    gb = b.groupby(["qubits", "model"]).agg(acc=("test_acc", "mean"), acc_s=("test_acc", "std"),
                                             t=("time_s", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.8))
    for m in colors:
        s = gb[gb.model == m]
        axes[0].errorbar(s.qubits, s.acc, yerr=s.acc_s, marker="o", ms=2.5, lw=1, capsize=1.5,
                         color=colors[m], label=m)
        axes[1].plot(s.qubits, s.t, marker="o", ms=2.5, lw=1, color=colors[m])
    axes[0].set_xlabel("Qubits (= composantes ACP)"); axes[0].set_ylabel("Accuracy (test)")
    axes[1].set_xlabel("Qubits"); axes[1].set_ylabel("Temps (s)"); axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xticks(sorted(b.qubits.unique()))
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=5, fontsize=6, loc="upper center", frameon=False, columnspacing=1.0, handlelength=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "fig_scaling.pdf"); fig.savefig(OUT / "fig_scaling.png", dpi=200)
    plt.close(fig)

    # C : couches du VQC
    c = df[df.experiment == "C"]
    gc = c.groupby("layers").agg(tr=("train_acc", "mean"), te=("test_acc", "mean"),
                                 te_s=("test_acc", "std")).reset_index()
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    ax.plot(gc.layers, gc.tr, "s--", ms=3, lw=1, color="#9b7fc1", label="entraînement")
    ax.errorbar(gc.layers, gc.te, yerr=gc.te_s, fmt="o-", ms=3, lw=1, capsize=1.5, color="#6a4c93", label="test")
    ax.set_xlabel("Couches $L$ du VQC"); ax.set_ylabel("Accuracy"); ax.set_xticks(gc.layers)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig_layers.pdf"); fig.savefig(OUT / "fig_layers.png", dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="version réduite (test rapide)")
    p.add_argument("--workers", type=int, default=None, help="processus parallèles (défaut : tous les cœurs)")
    p.add_argument("--plots-only", action="store_true", help="refait tableaux et figures depuis raw.csv")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        df = pd.read_csv(OUT / "raw.csv")
    else:
        jobs = build_jobs(args.quick)
        print(f"{len(jobs)} configurations à entraîner…")
        t0 = time.time()
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, res in enumerate(ex.map(run_job, jobs), 1):
                rows += res
                print(f"  {i}/{len(jobs)} terminées ({time.time() - t0:.0f} s)", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / "raw.csv", index=False)

    (OUT / "table_accuracy.tex").write_text(table_A(df), encoding="utf-8")
    (OUT / "table_cost.tex").write_text(table_cost(df), encoding="utf-8")
    figures(df)
    print(f"Résultats dans {OUT}")


if __name__ == "__main__":
    main()
