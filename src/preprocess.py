"""Özellikleri dönüştürür ve standartlaştırır. İstatistikler yalnızca eğitim verisinden alınır."""
import numpy as np


def signed_log1p(X: np.ndarray) -> np.ndarray:
    """Büyük değerleri sıkıştırır, negatif değerleri korur: sign(x) * log(1 + |x|)."""
    return np.sign(X) * np.log1p(np.abs(X))


class Standardizer:
    """Ortalama ve standart sapmayı yalnızca fit() ile verilen veriden öğrenir."""

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std == 0, 1.0, std)  # sabit sütunlarda sıfıra bölmeyi önler
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def stratified_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    """Sınıf oranını koruyarak eğitim/test bölmesi yapar."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_test = int(len(idx) * test_size)
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    train_idx = rng.permutation(np.concatenate(train_idx))
    test_idx = rng.permutation(np.concatenate(test_idx))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def preprocess(X_train: np.ndarray, X_test: np.ndarray):
    """Log dönüşümü ve standartlaştırmayı uygular. Dönüşüm yalnızca eğitim verisine göre öğrenilir."""
    X_train = signed_log1p(X_train)
    X_test = signed_log1p(X_test)
    scaler = Standardizer().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), scaler