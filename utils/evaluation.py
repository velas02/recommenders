"""
Evaluación reutilizada por todos los modelos que predicen una nota consultor-recurso
(CF, MF, MF con Keras...). No depende de cómo prediga cada modelo, solo de que
exponga una función `predecir_fn(consultor_id, recurso_id) -> nota`.
"""

from typing import Callable

import pandas as pd

from utils.config import K, RATING_RELEVANTE
from utils.metrics import rmse, mae, precision_at_k, hit_rate_at_k

PredecirFn = Callable[..., float]


def recomendar_top_k(
    consultor_id: int, candidatos: list[int], predecir_fn: PredecirFn, k: int = K
) -> list[int]:
    """Predice la nota de cada recurso candidato y devuelve los k mejores."""
    predicciones = {
        recurso_id: predecir_fn(consultor_id=consultor_id, recurso_id=recurso_id) for recurso_id in candidatos
    }
    ranking = sorted(predicciones, key=predicciones.get, reverse=True)
    return ranking[:k]


def evaluar(
    ratings_train: pd.DataFrame,
    ratings_test: pd.DataFrame,
    todos_los_recursos: list[int],
    predecir_fn: PredecirFn,
) -> tuple[float, float, float, float]:
    """
    RMSE/MAE de la nota predicha vs la real en test, precision@k del top-k
    recomendado, y hit_rate@k (¿acertamos al menos uno de los k?).
    """
    y_true = ratings_test["rating"].tolist()
    y_pred = [
        predecir_fn(consultor_id=fila.consultor_id, recurso_id=fila.recurso_id)
        for fila in ratings_test.itertuples()
    ]

    precisiones = []
    hit_rates = []
    for consultor_id in ratings_test["consultor_id"].unique():
        relevantes_test = ratings_test[
            (ratings_test["consultor_id"] == consultor_id) & (ratings_test["rating"] >= RATING_RELEVANTE)
        ]["recurso_id"].tolist()
        if not relevantes_test:
            continue

        ya_valorados = set(ratings_train.loc[ratings_train["consultor_id"] == consultor_id, "recurso_id"])
        candidatos = [r for r in todos_los_recursos if r not in ya_valorados]
        recomendados = recomendar_top_k(consultor_id=consultor_id, candidatos=candidatos, predecir_fn=predecir_fn)
        precisiones.append(precision_at_k(recomendados=recomendados, relevantes=relevantes_test, k=K))
        hit_rates.append(hit_rate_at_k(recomendados=recomendados, relevantes=relevantes_test, k=K))

    precision_media = sum(precisiones) / len(precisiones) if precisiones else 0.0
    hit_rate_media = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
    return rmse(y_true=y_true, y_pred=y_pred), mae(y_true=y_true, y_pred=y_pred), precision_media, hit_rate_media
