"""CIC-IDS2017 CSV dosyalarını yükler ve temizler."""
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COL = "Label"


def load_csv(path: str | Path) -> pd.DataFrame:
    """Tek bir CSV dosyasını okur, sütun adlarındaki baştaki/sondaki boşlukları siler."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sonsuz ve eksik değerli satırları atar, tekrarlanan satırları kaldırır."""
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df.drop_duplicates()


def binary_labels(df: pd.DataFrame) -> pd.Series:
    """BENIGN satırları 0, diğer tüm saldırı türlerini 1 yapar."""
    return (df[LABEL_COL].str.strip() != "BENIGN").astype(int)


def split_features(df: pd.DataFrame):
    """Özellik matrisi (X) ve ikili etiket vektörü (y) döndürür."""
    X = df.drop(columns=[LABEL_COL]).select_dtypes(include=np.number)
    y = binary_labels(df)
    return X, y