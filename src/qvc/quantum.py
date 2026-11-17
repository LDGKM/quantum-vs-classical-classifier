"""Modèles quantiques, écrits avec PennyLane.

Deux approches de l'apprentissage automatique quantique :

1. VariationalQuantumClassifier : un circuit à paramètres entraînables, optimisé
   par descente de gradient (l'analogue quantique d'un réseau de neurones).
2. QuantumKernelSVC : le circuit sert uniquement à mesurer la similarité entre
   deux points (un « noyau quantique »), puis un SVM classique fait la classification.

Les deux respectent l'interface scikit-learn (fit / predict / score).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC


def _make_device(n_qubits: int, device: str, shots):
    """Crée le simulateur (ou le vrai matériel, via un plugin PennyLane)."""
    return qml.device(device, wires=n_qubits, shots=shots)


class VariationalQuantumClassifier(BaseEstimator, ClassifierMixin):
    """Classifieur quantique variationnel (VQC).

    Circuit :  |0...0>  ->  encodage des données (rotations RY)
                        ->  couches entraînables (rotations + intrications CNOT)
                        ->  mesure de <Z> sur le qubit 0, dans [-1, 1]
    La classe prédite est le signe de <Z> + biais.
    """

    def __init__(
        self,
        n_layers: int = 2,
        steps: int = 60,
        lr: float = 0.1,
        batch_size: int = 16,
        device: str = "default.qubit",
        shots=None,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.n_layers = n_layers
        self.steps = steps
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.shots = shots
        self.seed = seed
        self.verbose = verbose

    # --- construction du circuit -------------------------------------------------
    def _build(self, n_qubits: int):
        dev = _make_device(n_qubits, self.device, self.shots)

        @qml.qnode(dev)
        def circuit(weights, x):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        self.circuit_ = circuit
        self.n_qubits_ = n_qubits

    def _raw_output(self, weights, bias, X):
        return pnp.stack([self.circuit_(weights, x) for x in X]) + bias

    # --- interface scikit-learn ---------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Ce classifieur ne gère que la classification binaire.")
        y_pm = np.where(y == self.classes_[1], 1.0, -1.0)  # étiquettes dans {-1, +1}

        self._build(X.shape[1])
        rng = np.random.default_rng(self.seed)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits_)
        weights = pnp.array(rng.uniform(0, 2 * np.pi, size=shape), requires_grad=True)
        bias = pnp.array(0.0, requires_grad=True)

        opt = qml.AdamOptimizer(stepsize=self.lr)

        def cost(w, b, Xb, yb):
            return pnp.mean((self._raw_output(w, b, Xb) - yb) ** 2)

        self.loss_history_ = []
        for step in range(self.steps):
            idx = rng.choice(len(X), size=min(self.batch_size, len(X)), replace=False)
            (weights, bias, _, _), loss = opt.step_and_cost(cost, weights, bias, X[idx], y_pm[idx])
            self.loss_history_.append(float(loss))
            if self.verbose and (step % 10 == 0 or step == self.steps - 1):
                print(f"  étape {step:3d} | perte = {float(loss):.4f}")

        self.weights_ = weights
        self.bias_ = bias
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return np.asarray(self._raw_output(self.weights_, self.bias_, X), dtype=float)

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])

    @property
    def n_params_(self) -> int:
        return int(np.size(self.weights_) + 1)

    def draw(self) -> str:
        """Dessin texte du circuit (pratique pour le README ou le rapport)."""
        x = np.zeros(self.n_qubits_)
        return qml.draw(self.circuit_, level="device", decimals=2)(self.weights_, x)


class QuantumKernelSVC(BaseEstimator, ClassifierMixin):
    """SVM dont le noyau est calculé par un circuit quantique.

    k(x, x') = |<phi(x')|phi(x)>|^2.

    Deux façons de le calculer :
    - mode "statevector" (par défaut en simulation exacte) : on simule l'état |phi(x)>
      de chaque point une seule fois, puis on fait des produits scalaires. Rapide :
      n circuits au lieu de n^2.
    - mode "circuit" (obligatoire sur vrai matériel ou avec shots) : pour chaque paire,
      on applique l'encodage de x puis l'inverse de celui de x', et on mesure la
      probabilité de revenir à |0...0>. C'est ce que ferait un vrai ordinateur quantique.
    """

    def __init__(
        self,
        C: float = 1.0,
        n_layers: int = 1,
        device: str = "default.qubit",
        shots=None,
        mode: str = "auto",
    ):
        self.C = C
        self.n_layers = n_layers
        self.device = device
        self.shots = shots
        self.mode = mode

    def _feature_map(self, x, wires):
        # Encodage en angles répété (« data re-uploading ») avec intrication :
        # rend le noyau plus riche qu'un simple produit de cosinus.
        for _ in range(self.n_layers):
            qml.AngleEmbedding(x, wires=wires, rotation="Y")
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
            qml.AngleEmbedding(x, wires=wires, rotation="Z")

    def _build(self, n_qubits: int):
        wires = list(range(n_qubits))
        mode = self.mode
        if mode == "auto":
            exact = self.shots is None and self.device.startswith("default.")
            mode = "statevector" if exact else "circuit"
        self.mode_ = mode

        if mode == "statevector":
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev)
            def state(x):
                self._feature_map(x, wires)
                return qml.state()

            self.state_ = state
        else:
            dev = _make_device(n_qubits, self.device, self.shots)

            @qml.qnode(dev)
            def overlap(x1, x2):
                self._feature_map(x1, wires)
                qml.adjoint(self._feature_map)(x2, wires)
                return qml.probs(wires=wires)

            self.kernel_ = lambda a, b: overlap(a, b)[0]

    def _states(self, X):
        return np.stack([np.asarray(self.state_(x)) for x in X])

    def _gram(self, A, B):
        if self.mode_ == "statevector":
            SA = self._states(A)
            SB = self.S_train_ if B is getattr(self, "X_train_", None) else self._states(B)
            return np.abs(SA.conj() @ SB.T) ** 2
        return np.asarray(qml.kernels.kernel_matrix(A, B, self.kernel_), dtype=float)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self._build(X.shape[1])
        self.X_train_ = X
        if self.mode_ == "statevector":
            self.S_train_ = self._states(X)
        self.svc_ = SVC(kernel="precomputed", C=self.C).fit(self._gram(X, X), y)
        self.classes_ = self.svc_.classes_
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.svc_.predict(self._gram(X, self.X_train_))
