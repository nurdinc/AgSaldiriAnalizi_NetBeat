"""PCA: kovaryans matrisinin özvektörleriyle boyut indirgeme (NumPy ile, sıfırdan)."""
import numpy as np


class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> "PCA":
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = (Xc.T @ Xc) / (len(X) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)  # küçükten büyüğe sıralı döner
        order = np.argsort(eigvals)[::-1]        # büyükten küçüğe çevir
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        self.components_ = eigvecs[:, : self.n_components]
        self.explained_variance_ratio_ = eigvals[: self.n_components] / eigvals.sum()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)