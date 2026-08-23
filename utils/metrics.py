"""Métricas reutilizadas para evaluar y comparar los distintos modelos de la demo.

- rmse / mae: para modelos que predicen una nota (CF, MF...), comparando la nota
  predicha contra la nota real dejada aparte en el test.
- precision_at_k: para modelos que devuelven un ranking de recursos (como TF-IDF),
  mide qué proporción de los k recomendados resultan ser relevantes de verdad.
- hit_rate_at_k: variante menos exigente de precision_at_k -en vez de "qué
  proporción de los k acierta", mide "¿acertamos al menos uno de los k?". Es
  la métrica habitual cuando cada consultor tiene pocos recursos relevantes
  en el test (protocolo "leave-one-out"), donde precision_at_k tiene un techo
  muy bajo aunque el modelo acierte todo lo que puede acertar.
"""

from typing import Sequence

import numpy as np


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_at_k(recomendados: list[int], relevantes: list[int], k: int = 5) -> float:
    """Proporción de los k primeros recomendados que están entre los relevantes."""
    top_k = recomendados[:k]
    if not top_k:
        return 0.0
    aciertos = len(set(top_k) & set(relevantes))
    return aciertos / len(top_k)


def hit_rate_at_k(recomendados: list[int], relevantes: list[int], k: int = 5) -> float:
    """1.0 si al menos uno de los k primeros recomendados está entre los relevantes, si no 0.0."""
    top_k = recomendados[:k]
    return 1.0 if set(top_k) & set(relevantes) else 0.0
