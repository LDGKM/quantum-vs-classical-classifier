"""Interface graphique : comparer classifieurs quantiques et classiques.

Lancement local :  streamlit run app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))  # _MEIPASS : exécutable PyInstaller
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from qvc import QuantumKernelSVC, VariationalQuantumClassifier, classical_models, load_dataset  # noqa: E402

st.set_page_config(page_title="Quantique vs classique", page_icon="⚛️", layout="wide")

QUANTUM = "#6a4c93"
CLASSIC = "#8d99ae"
DATASET_INFO = {
    "moons": ("Deux lunes", 2, "Deux croissants imbriqués : non linéaire mais simple."),
    "circles": ("Cercles", 2, "Un cercle dans un cercle : impossible à séparer avec une droite."),
    "iris": ("Iris (2 classes)", 4, "Fleurs setosa vs versicolor : très facile."),
    "breast_cancer": ("Cancer du sein", 30, "Tumeurs bénignes/malignes, 30 mesures réduites par ACP."),
}

# ------------------------------------------------------------------------------ barre latérale
st.sidebar.title("⚛️ Paramètres")
dataset = st.sidebar.selectbox(
    "Jeu de données", list(DATASET_INFO), format_func=lambda k: DATASET_INFO[k][0]
)
label, n_vars, desc = DATASET_INFO[dataset]
st.sidebar.caption(desc)

max_q = min(n_vars, 6)
n_qubits = (
    st.sidebar.slider("Nombre de qubits (= variables)", 2, max_q, min(4, max_q))
    if max_q > 2
    else 2
)
if max_q == 2:
    st.sidebar.caption("Ce jeu a 2 variables : 2 qubits.")

n_layers = st.sidebar.slider("Couches du circuit variationnel", 1, 4, 2)
steps = st.sidebar.slider("Itérations d'entraînement du VQC", 20, 150, 60, step=10)
n_samples = st.sidebar.slider("Nombre de points (jeux synthétiques)", 60, 400, 200, step=20,
                              disabled=dataset not in ("moons", "circles"))
seed = st.sidebar.number_input("Graine aléatoire", 0, 9999, 42)

ALL_MODELS = list(classical_models()) + ["VQC (quantique)", "SVM à noyau quantique"]
chosen = st.sidebar.multiselect("Modèles à comparer", ALL_MODELS, default=ALL_MODELS)
run = st.sidebar.button("🚀 Lancer la comparaison", type="primary", width="stretch")


# ------------------------------------------------------------------------------ calcul
def make_model(name: str):
    if name == "VQC (quantique)":
        return VariationalQuantumClassifier(n_layers=n_layers, steps=steps, seed=seed)
    if name == "SVM à noyau quantique":
        return QuantumKernelSVC()
    return classical_models(seed)[name]


def train_all(data, names):
    X_train, X_test, y_train, y_test = data
    rows, fitted = [], {}
    progress = st.progress(0.0, text="Entraînement…")
    for i, name in enumerate(names):
        progress.progress(i / len(names), text=f"Entraînement : {name}…")
        model = make_model(name)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        dt = time.perf_counter() - t0
        y_pred = model.predict(X_test)
        n_params = getattr(model, "n_params_", None)
        if hasattr(model, "coef_"):
            n_params = model.coef_.size + model.intercept_.size
        elif hasattr(model, "coefs_"):
            n_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
        rows.append({
            "Modèle": name,
            "Famille": "quantique" if "quantique" in name else "classique",
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "Temps (s)": dt,
            "Paramètres": n_params,
        })
        fitted[name] = model
    progress.empty()
    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    df["Paramètres"] = df["Paramètres"].astype("Int64")
    return df, fitted


# ------------------------------------------------------------------------------ page
st.title("Classifieurs quantiques vs classiques")
st.markdown(
    "Entraînez des modèles **quantiques** (simulés) et **classiques** sur les mêmes données, "
    "puis comparez précision, temps et nombre de paramètres. Réglez les paramètres à gauche, puis cliquez sur **Lancer**."
)

params = (dataset, n_qubits, n_layers, steps, n_samples, seed, tuple(chosen))
if run:
    if not chosen:
        st.warning("Choisissez au moins un modèle.")
    else:
        data = load_dataset(dataset, n_features=n_qubits, n_samples=n_samples, seed=seed)
        df, fitted = train_all(data, chosen)
        st.session_state.result = dict(params=params, data=data, df=df, fitted=fitted)

res = st.session_state.get("result")
tab_cmp, tab_data, tab_circ, tab_learn = st.tabs(
    ["📊 Comparaison", "🔵 Données", "⚛️ Circuit quantique", "📘 Comprendre"]
)

with tab_learn:
    st.markdown(r"""
### Les modèles
- **Régression logistique, SVM RBF, réseau de neurones** : les références classiques.
- **VQC (classifieur quantique variationnel)** : chaque variable devient l'angle d'une rotation sur un qubit,
  puis des couches de rotations *entraînables* et de portes CNOT (qui intriquent les qubits) transforment l'état.
  On mesure $\langle Z \rangle \in [-1, 1]$ sur le premier qubit : son signe donne la classe.
  Les angles sont ajustés par descente de gradient, comme les poids d'un réseau de neurones.
- **SVM à noyau quantique** : le circuit mesure la *ressemblance* entre deux points,
  $k(x, x') = |\langle\phi(x')|\phi(x)\rangle|^2$, puis un SVM classique classe à partir de ces ressemblances.

### Pourquoi le quantique ne gagne pas (encore)
Sur de petits problèmes, les modèles classiques sont très efficaces et extrêmement rapides.
Ici les circuits sont **simulés** sur un processeur classique : le coût double à chaque qubit ajouté.
Sur un vrai ordinateur quantique, chaque mesure est bruitée et doit être répétée des centaines de fois.
L'intérêt du VQC est ailleurs : il obtient souvent de bons résultats avec **très peu de paramètres**.

### À essayer
1. *Cercles* avec 1 couche, puis 3 couches : le VQC en a-t-il besoin ?
2. *Cancer du sein* de 2 à 6 qubits : la précision suit-elle ? Et le temps ?
3. Comparez les colonnes **Paramètres** du VQC et du réseau de neurones.
""")

if res is None:
    for tab in (tab_cmp, tab_data, tab_circ):
        with tab:
            st.info("Cliquez sur **🚀 Lancer la comparaison** dans la barre latérale.")
    st.stop()

if res["params"] != params:
    st.caption("⚠️ Les paramètres ont changé depuis le dernier calcul : cliquez sur **Lancer** pour mettre à jour.")

df, fitted = res["df"], res["fitted"]
X_train, X_test, y_train, y_test = res["data"]

with tab_cmp:
    best = df.iloc[0]
    q = df[df["Famille"] == "quantique"]
    c = df[df["Famille"] == "classique"]
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Meilleur modèle : {best['Modèle']}", f"{best['Accuracy']:.1%}")
    if len(q):
        m2.metric("Meilleur quantique", f"{q['Accuracy'].max():.1%}")
    if len(c):
        m3.metric("Meilleur classique", f"{c['Accuracy'].max():.1%}")

    shown = df.copy()
    shown["Paramètres"] = shown["Paramètres"].map(lambda v: "—" if pd.isna(v) else str(int(v)))
    st.dataframe(
        shown.style.format({"Accuracy": "{:.3f}", "F1": "{:.3f}", "Temps (s)": "{:.3f}"}),
        width="stretch", hide_index=True,
    )

    colors = [QUANTUM if f == "quantique" else CLASSIC for f in df["Famille"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].barh(df["Modèle"], df["Accuracy"], color=colors)
    axes[0].set_xlim(0, 1); axes[0].invert_yaxis(); axes[0].set_xlabel("Accuracy (test)")
    for i, v in enumerate(df["Accuracy"]):
        axes[0].text(v - 0.02, i, f"{v:.2f}", va="center", ha="right", fontsize=9, color="white")
    axes[1].barh(df["Modèle"], df["Temps (s)"].clip(lower=1e-3), color=colors)
    axes[1].set_xscale("log"); axes[1].invert_yaxis(); axes[1].set_xlabel("Temps d'entraînement (s, échelle log)")
    axes[1].set_yticklabels([])
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Violet : modèles quantiques (simulés) · Gris : modèles classiques")
    st.download_button("Télécharger les résultats (CSV)", df.to_csv(index=False).encode(),
                       file_name=f"{dataset}_{n_qubits}q.csv", mime="text/csv")

with tab_data:
    st.markdown(f"**{label}** — {len(X_train)} points d'entraînement, {len(X_test)} de test, "
                f"{X_train.shape[1]} variables mises à l'échelle dans [0, π] (une par qubit).")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(*X_train[:, :2].T, c=y_train, cmap="coolwarm", alpha=0.5, label="entraînement")
    ax.scatter(*X_test[:, :2].T, c=y_test, cmap="coolwarm", edgecolors="k", marker="s", label="test")
    ax.set_xlabel("variable 1 (angle du qubit 0)"); ax.set_ylabel("variable 2 (angle du qubit 1)")
    ax.legend(); fig.tight_layout()
    c1, _ = st.columns([3, 2])
    c1.pyplot(fig)
    if X_train.shape[1] > 2:
        st.caption("Seules les deux premières composantes principales sont affichées.")

with tab_circ:
    vqc = fitted.get("VQC (quantique)")
    if vqc is None:
        st.info("Ajoutez « VQC (quantique) » aux modèles pour voir son circuit.")
    else:
        st.markdown(f"**Circuit entraîné** — {vqc.n_qubits_} qubits, {n_layers} couche(s), "
                    f"{vqc.n_params_} paramètres. `RY` encode les données, `Rot` sont les rotations "
                    "entraînables, `●─X` sont les portes CNOT qui intriquent les qubits.")
        st.code(vqc.draw(), language=None)
        col_a, col_b = st.columns(2)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(vqc.loss_history_, color=QUANTUM)
        ax.set_xlabel("itération"); ax.set_ylabel("perte (erreur quadratique)")
        ax.set_title("Apprentissage du VQC"); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); col_a.pyplot(fig)
        if vqc.n_qubits_ == 2:
            g = np.linspace(0, np.pi, 25)
            xx, yy = np.meshgrid(g, g)
            zz = vqc.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            cs = ax.contourf(xx, yy, zz, levels=20, cmap="coolwarm", alpha=0.75)
            ax.scatter(*X_test.T, c=y_test, cmap="coolwarm", edgecolors="k")
            fig.colorbar(cs, ax=ax, label="⟨Z⟩ + biais")
            ax.set_title("Frontière de décision (test)")
            fig.tight_layout(); col_b.pyplot(fig)
        else:
            col_b.caption("La frontière de décision n'est affichée qu'avec 2 qubits.")
