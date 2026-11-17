"""Construit l'exécutable autonome pour le système courant (Windows, macOS ou Linux).

    pip install -r requirements-app.txt pyinstaller
    python packaging/build.py

Résultat dans dist/. PyInstaller ne sait pas « cross-compiler » : il faut lancer ce
script sur chaque système cible (c'est ce que fait .github/workflows/build-apps.yml).
"""

import os
import sys
from pathlib import Path

import PyInstaller.__main__

ROOT = Path(__file__).resolve().parents[1]
NAME = "QuantiqueVsClassique"
SEP = os.pathsep  # ";" sous Windows, ":" ailleurs

args = [
    str(ROOT / "packaging" / "launcher.py"),
    "--name", NAME,
    "--onefile",
    "--console",          # la fenêtre de terminal permet d'arrêter l'application
    "--noconfirm",
    "--clean",
    "--distpath", str(ROOT / "dist"),
    "--workpath", str(ROOT / "build" / "pyinstaller"),
    "--specpath", str(ROOT / "build"),
    "--add-data", f"{ROOT / 'app.py'}{SEP}.",
    "--add-data", f"{ROOT / 'src'}{SEP}src",
    # Streamlit et PennyLane chargent des fichiers et des plugins dynamiquement :
    "--collect-all", "streamlit",
    "--collect-all", "pennylane",
    "--copy-metadata", "streamlit",
    "--copy-metadata", "pennylane",
    "--collect-submodules", "sklearn",
    "--hidden-import", "qvc",
    "--paths", str(ROOT / "src"),
    # Bibliothèques lourdes inutiles ici, exclues si elles sont installées :
    "--exclude-module", "torch",
    "--exclude-module", "tensorflow",
    "--exclude-module", "jax",
    "--exclude-module", "qiskit",
    "--exclude-module", "IPython",
    "--exclude-module", "notebook",
]
if sys.platform == "darwin":
    args += ["--osx-bundle-identifier", "io.github.ldgkm.quantique-vs-classique"]

PyInstaller.__main__.run(args)
print(f"\nExécutable créé dans {ROOT / 'dist'}")
