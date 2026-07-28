"""
dqn.py contains the neural-network Q-function approximator.

Input: 42 board cells
Output: 7 Q-values
Architecture: 42 -> 256 -> 256 -> 256 -> 7
Hidden activation: ReLU
Output activation: linear
Loss: MSE on the selected action
Framework: CuPy
"""

import numpy as cpu_np
import cupy as np # for training on cuda
import random

random.seed(1337)
np.random.seed(1337)


class DQN():

    def __init__(self, lr=0.001):

        self.lr = np.float32(lr)

        # He initialization for ReLU layers
        self.W1 = (np.random.randn(256, 42) * np.sqrt(2 / 42)).astype(np.float32)
        self.b1 = np.zeros((256, 1), dtype=np.float32)

        self.W2 = (np.random.randn(256, 256) * np.sqrt(2 / 256)).astype(np.float32)
        self.b2 = np.zeros((256, 1), dtype=np.float32)

        self.W3 = (np.random.randn(256, 256) * np.sqrt(2 / 256)).astype(np.float32)
        self.b3 = np.zeros((256, 1), dtype=np.float32)

        # Xavier initialization for linear output
        self.W4 = (np.random.randn(7, 256) * np.sqrt(1 / 256)).astype(np.float32)
        self.b4 = np.zeros((7, 1), dtype=np.float32)
    
    def ReLu(self, Z):
        return np.maximum(0, Z)


    """ Returns Z1, h1, Z2, h2, Z3, h3, Z4, y """
    def forward(self, x):
        
        Z1 = self.W1 @ x + self.b1
        h1 = self.ReLu(Z1)

        Z2 = self.W2 @ h1 + self.b2
        h2 = self.ReLu(Z2)

        Z3 = self.W3 @ h2 + self.b3
        h3 = self.ReLu(Z3)

        Q = self.W4 @ h3 + self.b4

        return Z1, h1, Z2, h2, Z3, h3, Q

    def backward(self, Z1, h1, Z2, h2, Z3, h3, Q, X, actions, targets):
        batch_size = X.shape[1]

        Q_bar = np.zeros_like(Q)

        batch_indices = np.arange(batch_size)

        Q_bar[actions, batch_indices] = (Q[actions, batch_indices] - targets)

        W4_bar = (Q_bar @ h3.T) / batch_size
        b4_bar = np.mean(Q_bar, axis=1, keepdims=True)

        h3_bar = self.W4.T @ Q_bar
        Z3_bar = h3_bar * (Z3 > 0)

        W3_bar = (Z3_bar @ h2.T) / batch_size
        b3_bar = np.mean(Z3_bar, axis=1, keepdims=True)

        h2_bar = self.W3.T @ Z3_bar
        Z2_bar = h2_bar * (Z2 > 0)

        W2_bar = (Z2_bar @ h1.T) / batch_size
        b2_bar = np.mean(Z2_bar, axis=1, keepdims=True)

        h1_bar = self.W2.T @ Z2_bar
        Z1_bar = h1_bar * (Z1 > 0)

        W1_bar = (Z1_bar @ X.T) / batch_size
        b1_bar = np.mean(Z1_bar, axis=1, keepdims=True)

        return (
            W1_bar, b1_bar,
            W2_bar, b2_bar,
            W3_bar, b3_bar,
            W4_bar, b4_bar
        )

    def update_params(self, W1_bar, b1_bar, W2_bar, b2_bar, W3_bar, b3_bar, W4_bar, b4_bar):
        self.W1 = self.W1 - self.lr * W1_bar
        self.b1 = self.b1 - self.lr * b1_bar

        self.W2 = self.W2 - self.lr * W2_bar
        self.b2 = self.b2 - self.lr * b2_bar

        self.W3 = self.W3 - self.lr * W3_bar
        self.b3 = self.b3 - self.lr * b3_bar

        self.W4 = self.W4 - self.lr * W4_bar
        self.b4 = self.b4 - self.lr * b4_bar
    
    def predict(self, X):
        *_, Q = self.forward(X)
        return Q

    def train_batch(self, X, actions, targets):

        X = np.asarray(X, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int32)
        targets = np.asarray(targets, dtype=np.float32)

        Z1, h1, Z2, h2, Z3, h3, Q = self.forward(X)

        batch_indices = np.arange(X.shape[1])
        chosen_q = Q[actions, batch_indices]

        loss = np.mean(0.5 * (chosen_q - targets) ** 2)

        gradients = self.backward(
            Z1, h1,
            Z2, h2,
            Z3, h3,
            Q,
            X,
            actions,
            targets
        )

        self.update_params(*gradients)

        return float(loss)

    def save_weights(self, filename="model_weights.npz", **metadata):
        save_data = {
            "W1": np.asnumpy(self.W1),
            "b1": np.asnumpy(self.b1),
            "W2": np.asnumpy(self.W2),
            "b2": np.asnumpy(self.b2),
            "W3": np.asnumpy(self.W3),
            "b3": np.asnumpy(self.b3),
            "W4": np.asnumpy(self.W4),
            "b4": np.asnumpy(self.b4),
        }

        # Add values such as episode, epsilon, and training_steps.
        save_data.update(metadata)

        cpu_np.savez_compressed(
            filename,
            **save_data
        )

        print(f"Weights saved to {filename}")

    def load_weights(self, filename="model_weights.npz"):
        with cpu_np.load(filename) as data:
            self.W1 = np.asarray(data["W1"], dtype=np.float32)
            self.b1 = np.asarray(data["b1"], dtype=np.float32)

            self.W2 = np.asarray(data["W2"], dtype=np.float32)
            self.b2 = np.asarray(data["b2"], dtype=np.float32)

            self.W3 = np.asarray(data["W3"], dtype=np.float32)
            self.b3 = np.asarray(data["b3"], dtype=np.float32)

            self.W4 = np.asarray(data["W4"], dtype=np.float32)
            self.b4 = np.asarray(data["b4"], dtype=np.float32)

            metadata = {}

            if "episode" in data.files:
                metadata["episode"] = int(data["episode"])

            if "epsilon" in data.files:
                metadata["epsilon"] = float(data["epsilon"])

            if "training_steps" in data.files:
                metadata["training_steps"] = int(
                    data["training_steps"]
                )

        print(f"Weights loaded from {filename}")

        return metadata

    def copy_weights_from(self, other):
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()

        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()

        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()

        self.W4 = other.W4.copy()
        self.b4 = other.b4.copy()


# model = DQN(0.0003)

# weights_file = "dqn_weights.npz"

# if os.path.exists(weights_file):
#     model.load_weights(weights_file)

# model.gradient_descent(D_train, D_test, 100)
# model.save_weights(weights_file)
