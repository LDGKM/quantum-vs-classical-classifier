# quantum-vs-classical-classifier

**Un classifieur quantique fait-il mieux qu'un classifieur classique ?**
Ce projet compare, sur les mêmes données et avec les mêmes métriques :

| Famille | Modèles |
|---|---|
| Classique | Régression logistique, SVM (noyau RBF), petit réseau de neurones |
| Quantique | Classifieur quantique variationnel (VQC), SVM à noyau quantique |

Les modèles quantiques sont écrits avec [PennyLane](https://pennylane.ai) et respectent l'interface scikit-learn (`fit` / `predict` / `score`).
Ils tournent en simulation, puis (phase 4) sur un vrai processeur IBM Quantum.

Projet 1 sur 4 d'une série IA × informatique quantique :
[`qaoa-ml-optimizer`](https://github.com/LDGKM/qaoa-ml-optimizer) ·
[`quantum-circuit-agent`](https://github.com/LDGKM/quantum-circuit-agent) ·
[`neural-quantum-decoder`](https://github.com/LDGKM/neural-quantum-decoder)

---

## Utiliser l'application (rien à installer)

| | Comment |
|---|---|
| 🌐 **En ligne** | Ouvrir le lien du Space Hugging Face : `https://huggingface.co/spaces/<pseudo-hf>/quantum-vs-classical-classifier` |
| 🪟 🍎 🐧 **Hors ligne** | Télécharger le fichier de son système dans les [Releases](https://github.com/LDGKM/quantum-vs-classical-classifier/releases), le dézipper et le lancer |

L'application permet de choisir le jeu de données, le nombre de qubits et de couches, puis affiche
le tableau comparatif, les graphiques, le circuit quantique entraîné et sa frontière de décision.

> Windows peut afficher « Windows a protégé votre ordinateur » (l'exécutable n'est pas signé) :
> *Informations complémentaires* → *Exécuter quand même*. Sur Mac : clic droit → *Ouvrir* la première fois.

### Mise en ligne (une seule fois, pour le mainteneur)

**Application web.** Le workflow `.github/workflows/deploy-hf-space.yml` publie l'application sur Hugging Face
à chaque push sur `main`.
1. Sur huggingface.co : *Settings* → *Access Tokens* → *Create new token*, type **Write**. Copier le jeton.
2. Sur GitHub, dans ce dépôt : *Settings* → *Secrets and variables* → *Actions* :
   - onglet **Secrets** → *New repository secret* : nom `HF_TOKEN`, valeur = le jeton ;
   - onglet **Variables** → *New repository variable* : nom `HF_USERNAME`, valeur = votre pseudo Hugging Face.
3. Pousser sur `main` (ou *Actions* → *Déployer sur Hugging Face* → *Run workflow*). Le Space est créé automatiquement ;
   la première construction prend environ 5 minutes.

**Exécutables.** Le workflow `.github/workflows/build-apps.yml` construit l'application sur Windows, macOS et Linux
(un exécutable ne fonctionne que sur le système où il a été construit), vérifie qu'elle démarre, puis la publie :
```bash
git tag v0.1.0
git push origin v0.1.0
```

Lancer l'interface depuis le code source : `streamlit run app.py` (après `setup.bat`).

---

## Installation (une seule commande)

Prérequis : **Python 3.10 à 3.13** ([python.org](https://www.python.org/downloads/), cocher *Add python.exe to PATH*).

| Système | Commande |
|---|---|
| Windows | double-cliquer sur **`setup.bat`** |
| Linux / macOS / Git Bash | `bash setup.sh` |

Le script crée un environnement virtuel `.venv` dans le dossier du projet, y installe toutes les dépendances
(`requirements.txt`) et enregistre le noyau Jupyter **Python (quantum-classifier)**.
Rien n'est installé ailleurs sur la machine ; pour tout désinstaller, il suffit de supprimer `.venv`.

> Le dossier `.venv` n'est pas versionné (il est dans `.gitignore`) : un environnement virtuel dépend du système
> et du chemin où il a été créé, il ne peut pas être copié d'une machine à l'autre. Chaque utilisateur le recrée avec le script.

## Utilisation

```bash
.venv\Scripts\activate            # Windows   (Linux/macOS : source .venv/bin/activate)

python -m pytest                  # vérifie l'installation (~15 s)
python run_benchmark.py           # comparaison sur "moons", 2 qubits (~15 s)
python run_benchmark.py --dataset breast_cancer --qubits 4 --steps 80
jupyter lab                       # ouvre les notebooks
```

Chaque benchmark écrit dans `results/` un tableau CSV, un tableau LaTeX (inclus automatiquement dans le rapport) et un graphique.

Options de `run_benchmark.py` : `--dataset {moons,circles,iris,breast_cancer}`, `--qubits`, `--layers`, `--steps`, `--samples`, `--seed`, `--no-kernel`.

### Premier résultat (moons, 2 qubits, simulation exacte)

| Modèle | Accuracy | Temps d'entraînement | Paramètres |
|---|---|---|---|
| SVM (noyau RBF) | 0,983 | 0,001 s | — |
| Petit réseau de neurones | 0,983 | 0,5 s | 337 |
| SVM à noyau quantique | 0,950 | 0,2 s | — |
| VQC (quantique) | 0,917 | 9,5 s | **13** |
| Régression logistique | 0,900 | 0,002 s | 3 |

Lecture : sur un petit problème, le quantique **ne bat pas** le classique, mais le VQC atteint 92 % avec 26 fois moins
de paramètres que le réseau de neurones. Toute la question du projet est de savoir si cet écart se réduit, et à quel coût.

## Structure

```
├── app.py                      interface graphique (Streamlit)
├── setup.bat / setup.sh        installation en une commande (développement)
├── requirements.txt            dépendances de développement
├── requirements-app.txt        dépendances de l'application seule
├── run_benchmark.py            comparaison en ligne de commande
├── Dockerfile, space/          application web sur Hugging Face Spaces
├── packaging/                  construction des exécutables (PyInstaller)
├── .github/workflows/          déploiement web et exécutables automatiques
├── src/qvc/
│   ├── data.py                 jeux de données → ACP → angles dans [0, π]
│   ├── classical.py            modèles de référence
│   ├── quantum.py              VQC et SVM à noyau quantique
│   └── benchmark.py            entraînement, métriques, export CSV/LaTeX/PNG
├── notebooks/
│   └── 01_premiers_qubits.ipynb   qubits, mesure, intrication, premier VQC
├── tests/test_smoke.py
├── docs/rapport.tex            rapport du projet (LaTeX)
└── results/                    sorties des benchmarks
```

## Feuille de route

- [x] **Phase 1 — Bases** : qubits, mesure, intrication, encodage (notebook 01)
- [x] **Phase 2 — Modèles** : VQC, noyau quantique, références classiques, benchmark
- [ ] **Phase 3 — Étude systématique** : faire varier qubits, couches, taille des données ; tracer accuracy et temps
- [ ] **Phase 4 — Bruit et vrai matériel** : simulation avec shots et bruit, puis exécution sur IBM Quantum (`qiskit-ibm-runtime`, compte gratuit)
- [ ] **Phase 5 — Rapport** : compléter `docs/rapport.tex` avec les résultats et l'analyse

## Compiler le rapport

```bash
cd docs
pdflatex rapport.tex && pdflatex rapport.tex
```
(ou importer le dossier `docs/` et `results/` dans Overleaf). Le rapport inclut automatiquement `results/moons_2q.tex` s'il existe.

## Licence

MIT
