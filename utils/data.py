"""Carga y partición del dataset, reutilizada por todos los experimentos."""

from pathlib import Path

import pandas as pd

DATASET_DIR = Path(__file__).parent.parent / "dataset"


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lee los tres CSV del dataset sintético y los devuelve como DataFrames."""
    consultores = pd.read_csv(DATASET_DIR / "consultores.csv")
    recursos = pd.read_csv(DATASET_DIR / "recursos.csv")
    ratings = pd.read_csv(DATASET_DIR / "ratings.csv")
    return consultores, recursos, ratings


def train_test_split_ratings(
    ratings: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baraja los ratings y los separa en train/test (split simple, sin estratificar)."""
    ratings = ratings.sample(frac=1, random_state=seed).reset_index(drop=True)
    corte = int(len(ratings) * (1 - test_size))
    return ratings.iloc[:corte], ratings.iloc[corte:]


def construir_matriz(ratings: pd.DataFrame) -> pd.DataFrame:
    """Tabla consultor x recurso con la nota de cada uno (NaN si no lo ha valorado)."""
    return ratings.pivot_table(index="consultor_id", columns="recurso_id", values="rating")


def indexar_ids(valores: pd.Series) -> dict[int, int]:
    """Mapea cada id real a una posición 0..n-1, la que esperan los Embedding de Keras."""
    return {id_: i for i, id_ in enumerate(valores)}
