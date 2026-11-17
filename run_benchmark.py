"""Lance la comparaison quantique vs classique depuis la ligne de commande.

Exemples :
    python run_benchmark.py
    python run_benchmark.py --dataset breast_cancer --qubits 4 --layers 3 --steps 100
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from qvc import DATASETS, run_benchmark  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Classifieurs quantiques vs classiques")
    p.add_argument("--dataset", choices=DATASETS, default="moons")
    p.add_argument("--qubits", type=int, default=2, help="nombre de qubits = nombre de variables")
    p.add_argument("--layers", type=int, default=2, help="couches du circuit variationnel")
    p.add_argument("--steps", type=int, default=60, help="itérations d'optimisation du VQC")
    p.add_argument("--samples", type=int, default=200, help="taille des jeux synthétiques")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-kernel", action="store_true", help="ignore le SVM à noyau quantique (plus lent)")
    args = p.parse_args()

    df = run_benchmark(
        dataset=args.dataset,
        n_qubits=args.qubits,
        n_layers=args.layers,
        steps=args.steps,
        n_samples=args.samples,
        seed=args.seed,
        include_kernel=not args.no_kernel,
    )
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nRésultats enregistrés dans le dossier results/")


if __name__ == "__main__":
    main()
