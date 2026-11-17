"""Tests rapides : vérifient que tout s'installe et tourne de bout en bout."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qvc import QuantumKernelSVC, VariationalQuantumClassifier, load_dataset, run_benchmark  # noqa: E402


def test_dataset_shapes_and_range():
    X_train, X_test, y_train, y_test = load_dataset("breast_cancer", n_features=3)
    assert X_train.shape[1] == X_test.shape[1] == 3
    assert X_train.min() >= 0 and X_train.max() <= np.pi + 1e-9
    assert set(np.unique(y_train)) == {0, 1}


def test_vqc_learns_something():
    X_train, X_test, y_train, y_test = load_dataset("iris", n_features=2)
    clf = VariationalQuantumClassifier(n_layers=2, steps=40, seed=1).fit(X_train, y_train)
    assert clf.loss_history_[-1] < clf.loss_history_[0]
    assert clf.score(X_test, y_test) > 0.7  # iris setosa/versicolor est facile


def test_quantum_kernel_runs():
    X_train, X_test, y_train, y_test = load_dataset("moons", n_samples=40)
    clf = QuantumKernelSVC().fit(X_train, y_train)
    assert clf.predict(X_test).shape == y_test.shape


def test_benchmark_table(tmp_path):
    df = run_benchmark("moons", n_samples=40, steps=5, verbose=False, out_dir=tmp_path)
    assert len(df) == 5
    assert (tmp_path / "moons_2q.csv").exists()
    assert (tmp_path / "moons_2q.tex").exists()


def test_kernel_modes_agree():
    """Les deux méthodes de calcul du noyau doivent donner la même matrice."""
    X_train, *_ = load_dataset("moons", n_samples=20)
    A = QuantumKernelSVC(mode="statevector")
    A._build(2)
    B = QuantumKernelSVC(mode="circuit")
    B._build(2)
    assert np.allclose(A._gram(X_train[:5], X_train[:5]), B._gram(X_train[:5], X_train[:5]), atol=1e-6)
