"""Mahalanobis anomali dedektörü: yalnızca normal (BENIGN) veriyle eğitilir."""
import numpy as np


class MahalanobisDetector:
    def __init__(self, reg: float = 1e-6):
        self.reg = reg  # kovaryans matrisinin tersinin alınabilmesi için eklenen küçük değer

    def fit(self, X_normal: np.ndarray) -> "MahalanobisDetector":
        self.mu_ = X_normal.mean(axis=0)
        Xc = X_normal - self.mu_
        cov = (Xc.T @ Xc) / (len(X_normal) - 1)
        cov += self.reg * np.eye(cov.shape[0])
        self.precision_ = np.linalg.inv(cov)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Her örnek için Mahalanobis uzaklığının karesi. Büyük değer daha anormal demektir."""
        D = X - self.mu_
        return np.einsum("ij,jk,ik->i", D, self.precision_, D)