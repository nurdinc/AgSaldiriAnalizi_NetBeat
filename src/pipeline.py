"""Uçtan uca akış: veri -> bölme -> log/standartlaştırma -> PCA -> Mahalanobis -> eşik -> değerlendirme."""
import numpy as np
from scipy.stats import rankdata

from src.data import load_all, split_features
from src.preprocess import stratified_split, signed_log1p, Standardizer
from src.pca import PCA
from src.mahalanobis import MahalanobisDetector


def roc_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney U tabanlı ROC-AUC: bir saldırının normalden daha yüksek puan alma olasılığı."""
    ranks = rankdata(scores)
    n_pos = np.sum(y == 1)
    n_neg = len(y) - n_pos
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def best_threshold(scores: np.ndarray, y: np.ndarray):
    """Doğrulama verisinde F1 skorunu en yüksek yapan eşiği bulur."""
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 501)))
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        pred = scores >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        fn = np.sum(~pred & (y == 1))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def run(folder: str = "data/MachineLearningCVE", n_components: int = 20, seed: int = 42):
    df = load_all(folder)
    X, y = split_features(df)
    X = X.to_numpy(dtype=float)
    y = y.to_numpy()

    # %60 eğitim, %20 doğrulama, %20 test
    X_rest, X_test, y_rest, y_test = stratified_split(X, y, test_size=0.2, seed=seed)
    X_train, X_val, y_train, y_val = stratified_split(X_rest, y_rest, test_size=0.25, seed=seed)
    del X, X_rest, y_rest

    # Log dönüşümü ve standartlaştırma: istatistikler yalnızca eğitimden
    X_train = signed_log1p(X_train)
    X_val = signed_log1p(X_val)
    X_test = signed_log1p(X_test)
    scaler = Standardizer().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # PCA: yalnızca eğitim verisine göre
    pca = PCA(n_components).fit(X_train)
    Z_train = pca.transform(X_train)
    Z_val = pca.transform(X_val)
    Z_test = pca.transform(X_test)

    # Mahalanobis: yalnızca eğitimdeki normal (BENIGN) trafik
    detector = MahalanobisDetector().fit(Z_train[y_train == 0])
    s_val = detector.score(Z_val)
    s_test = detector.score(Z_test)

    # Eşik doğrulamada seçilir, test verisine olduğu gibi uygulanır
    t, f1_val = best_threshold(s_val, y_val)
    pred = s_test >= t
    tp = np.sum(pred & (y_test == 1))
    fp = np.sum(pred & (y_test == 0))
    fn = np.sum(~pred & (y_test == 1))

    print(f"Eğitim: {len(y_train)}  Doğrulama: {len(y_val)}  Test: {len(y_test)}")
    print(f"PCA bileşen sayısı: {n_components}, açıklanan varyans: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"Doğrulama F1 (eşik seçimi): {f1_val:.3f}")
    print(f"Test ROC-AUC: {roc_auc(s_test, y_test):.3f}")
    print(f"Test precision: {tp / max(tp + fp, 1):.3f}  recall: {tp / max(tp + fn, 1):.3f}")


if __name__ == "__main__":
    run()