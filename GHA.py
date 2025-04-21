import numpy as np

class GHA:
    def __init__(self, input_dim, num_components, learning_rate=0.001):
        self.m = input_dim           # Input vector size (64)
        self.l = num_components      # Number of principal components (e.g., 8, 16, ...)
        self.eta = learning_rate     # Learning rate

        # Initialize weights: W shape (l, m) = (num_components, input_dim)
        self.W = np.random.randn(self.l, self.m) * 0.01

    def train(self, X, epochs=100):
        for epoch in range(epochs):
            np.random.shuffle(X)  # shuffle for better convergence
            for x in X:
                x = x.reshape(-1, 1)             # Shape: (64, 1)
                y = self.W @ x                   # Shape: (k, 1)

                # Outer product: y @ x.T -> (k, 64)
                # Lower-triangular part of y @ y.T -> (k, k)
                delta_W = self.eta * ((y @ x.T) - np.tril(y @ y.T) @ self.W)
                self.W += delta_W

            if np.any(np.isnan(self.W)):
                print(f"❌ NaNs detected at epoch {epoch}, stopping early. Check if inputs are normalized.")
                break

            self.eta *= 0.99
        
        self.W = self._normalize_components()

    def _normalize_components(self):
        norms = np.linalg.norm(self.W, axis=1, keepdims=True)
        return self.W / norms

    def get_components(self):
        return self.W.copy()  # shape: (l, m)
