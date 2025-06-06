import numpy as np

class GHA:
    def __init__(self, input_dim, num_components):
        self.m = input_dim
        self.l = num_components
        self.W = np.random.randn(self.l, self.m) * 0.01
        self.W -= np.mean(self.W, axis=1, keepdims=True)

    def train_parallel(self, X, epochs=100, lr_s=0.001, lr_f=0.0001):
        for epoch in range(epochs):
            np.random.shuffle(X)

            lr = lr_s * ((lr_f / lr_s) ** (epoch / epochs-1))

            if epoch % 1000 == 0:
                print(f"- Epoch {epoch}/{epochs}")

            for x in X: 
                x = x.reshape(-1, 1)
                y = self.W @ x
                delta_W = lr * ((y @ x.T) - np.tril(y @ y.T) @ self.W)
                self.W += delta_W

            if np.any(np.isnan(self.W)):
                print(f"NaNs detected at epoch {epoch}, stopping early. Check if inputs are normalized.")
                break

        
        self.W /= np.linalg.norm(self.W, axis=1, keepdims=True)

    
    def train_sequential(self, X, epochs_per_component=100, lr_s=0.001, lr_f=0.0001):
        components = []
        residual = X.copy()

        for i in range(self.l):
            if i % 2 == 0:
                print(f"- Sequential training: component {i+1}/{self.l}")
            

            gha_1 = GHA(input_dim=self.m, num_components=1)
            gha_1._train_one_component(residual, epochs_per_component, lr_s, lr_f)
            w = gha_1.get_components()[0]
            components.append(w)

            # deflation
            projections = residual @ w
            residual -= np.outer(projections, w)

        self.W = np.vstack(components)
        self.W /= np.linalg.norm(self.W, axis=1, keepdims=True)

    def _train_one_component(self, X, epochs=100, lr_s=0.001, lr_f=0.0001):
        for epoch in range(epochs):
            
            if epoch % 100 == 0:
                print(f"- Epoch {epoch}/{epochs}")

            np.random.shuffle(X)

            lr = lr_s * ((lr_f / lr_s) ** (epoch / epochs-1))

            for x in X:
                x = x.reshape(-1, 1)
                y = self.W @ x
                delta_W = lr * y * (x.T - y * self.W)
                self.W += delta_W

            if np.any(np.isnan(self.W)):
                print(f"NaNs detected at epoch {epoch}, stopping.")
                break


    def get_components(self):
        return self.W.copy()
