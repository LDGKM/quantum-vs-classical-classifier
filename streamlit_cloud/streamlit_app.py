"""Point d'entrée pour Streamlit Community Cloud.

Ce dossier existe pour une seule raison : Streamlit Cloud installe le requirements.txt
situé à côté du point d'entrée. Celui-ci ne contient que les dépendances de
l'application (plus léger et plus rapide que le requirements.txt de développement).
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "app.py"), run_name="__main__")
