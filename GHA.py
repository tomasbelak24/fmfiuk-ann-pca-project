import numpy as np

class GHA:
    def __init__(self, input_dim, num_components, learning_rate=0.001):
        self.m = input_dim           # Input vector size (64)
        self.l = num_components      # Number of principal components
        self.eta = learning_rate     # Learning rate
        self.W = np.random.randn(self.l, self.m) * 0.01

    def train_parallel(self, X, epochs=100):
        for epoch in range(epochs):
            np.random.shuffle(X)
            for x in X: 
                x = x.reshape(-1, 1)
                y = self.W @ x
                delta_W = self.eta * ((y @ x.T) - np.tril(y @ y.T) @ self.W)
                self.W += delta_W

            if np.any(np.isnan(self.W)):
                print(f"❌ NaNs detected at epoch {epoch}, stopping early. Check if inputs are normalized.")
                break

            self.eta *= 0.99
        
        #self.W /= np.linalg.norm(self.W, axis=1, keepdims=True)

    def train_sequential(self, X, epochs=100, tolerance=1e-5):
        components = []
        residual = X.copy()

        for i in range(self.l):
            print(f"- Sequential training: component {i+1}/{self.l}")
            # 1-component GHA
            gha_1 = GHA(input_dim=self.m, num_components=1, learning_rate=self.eta)
            gha_1.train(residual, epochs)
            w = gha_1.get_components()[0]
            w /= np.linalg.norm(w)  # normalize manually
            components.append(w)

            # Project out the learned component
            projections = residual @ w
            residual -= np.outer(projections, w)

        #self.W = np.vstack(components)

    def get_components(self):
        return self.W.copy()
