"""
Paso 2 de la demo: filtrado colaborativo (CF) basado en memoria.

Recomienda a partir de los patrones de interacción entre consultores y
recursos, sin mirar la metadata de estos (a diferencia del paso 1). Incluye
las dos variantes clásicas:
  - user-based: consultores parecidos entre sí -> lo que les gustó a unos
    se recomienda a los demás.
  - item-based: recursos parecidos entre sí, según a quién los ha valorado
    de forma parecida.

En ambos casos, la nota que un consultor daría a un recurso se predice como
su propia media más la desviación media (respecto a su propia media) de sus 
vecinos más afines, ponderada por esa afinidad.

Nota sobre densidad: el CF de memoria necesita suficientes coincidencias
entre dos consultores (o dos recursos) para que su similitud sea fiable; con
pocos datos en común, la similitud deja de ser representativa y el modelo
pierde precisión. Por eso se deja siempre en el log un baseline ingenuo
(predecir la media) como referencia: un CF que no lo supere no está
aportando valor real.

Ejecución (desde recommenders/):
    uv run scripts/02_cf.py
"""

import pandas as pd

from utils.config import K
from utils.data import cargar_datos, train_test_split_ratings, construir_matriz
from utils.evaluation import recomendar_top_k, evaluar
from utils.metrics import rmse, mae
from utils.logging_utils import get_logger

log = get_logger(nombre="02_cf")

MIN_COINCIDENCIAS = 3  # mínimo de valoraciones en común para que una similitud cuente


def predecir(
    fila_id: int,
    columna_id: int,
    matriz: pd.DataFrame,
    similitudes: pd.DataFrame,
    medias: pd.Series,
    media_global: float,
) -> float:
    """
    Predice matriz[fila_id, columna_id] como la media de esa fila más la
    desviación media (respecto a su propia media) de las demás filas que sí
    tienen valor en esa columna, ponderada por su similitud con fila_id.

    Sirve tanto para CF user-based (filas=consultores, columnas=recursos)
    como para item-based (filas=recursos, columnas=consultores, pasando la
    matriz traspuesta) -misma fórmula, solo cambia qué eje es cuál.
    """
    media_fila = medias.get(fila_id, media_global)
    if fila_id not in matriz.index or columna_id not in matriz.columns:
        return media_fila

    vecinos = matriz[columna_id].dropna().drop(labels=[fila_id], errors="ignore")
    if vecinos.empty:
        return media_fila

    pesos = similitudes.loc[fila_id, vecinos.index].dropna()
    if pesos.empty or pesos.abs().sum() == 0:
        return media_fila
    vecinos = vecinos.loc[pesos.index]

    desviaciones = vecinos - medias.loc[pesos.index]
    prediccion = media_fila + (pesos * desviaciones).sum() / pesos.abs().sum()
    return min(5, max(1, prediccion))


def main() -> None:
    consultores, recursos, ratings = cargar_datos()
    ratings_train, ratings_test = train_test_split_ratings(ratings=ratings)
    todos_los_recursos = recursos["recurso_id"].tolist()

    matriz = construir_matriz(ratings=ratings_train)
    media_global = ratings_train["rating"].mean()
    medias_consultor = matriz.mean(axis=1)
    medias_recurso = matriz.mean(axis=0)
    similitud_consultores = matriz.T.corr(min_periods=MIN_COINCIDENCIAS)
    similitud_recursos = matriz.corr(min_periods=MIN_COINCIDENCIAS)

    def predecir_user_based(consultor_id: int, recurso_id: int) -> float:
        return predecir(
            fila_id=consultor_id,
            columna_id=recurso_id,
            matriz=matriz,
            similitudes=similitud_consultores,
            medias=medias_consultor,
            media_global=media_global,
        )

    def predecir_item_based(consultor_id: int, recurso_id: int) -> float:
        return predecir(
            fila_id=recurso_id,
            columna_id=consultor_id,
            matriz=matriz.T,
            similitudes=similitud_recursos,
            medias=medias_recurso,
            media_global=media_global,
        )

    log.info(
        "Consultores: %d | Recursos: %d | Ratings train/test: %d/%d",
        len(consultores), len(recursos), len(ratings_train), len(ratings_test),
    )

    # Baseline ingenuo (predecir siempre la media global) para tener una referencia
    # honesta: un CF que no le gane a esto no está aportando nada de verdad.
    baseline_pred = [media_global] * len(ratings_test)
    log.info(
        "Baseline (media global) -> RMSE: %.3f | MAE: %.3f",
        rmse(y_true=ratings_test["rating"], y_pred=baseline_pred),
        mae(y_true=ratings_test["rating"], y_pred=baseline_pred),
    )

    for nombre, predecir_fn in [("user-based", predecir_user_based), ("item-based", predecir_item_based)]:
        rmse_, mae_, precision, hit_rate = evaluar(
            ratings_train=ratings_train,
            ratings_test=ratings_test,
            todos_los_recursos=todos_los_recursos,
            predecir_fn=predecir_fn,
        )
        log.info(
            "CF %s -> RMSE: %.3f | MAE: %.3f | Precision@%d: %.3f | Hit Rate@%d: %.3f",
            nombre, rmse_, mae_, K, precision, K, hit_rate,
        )

    # Ejemplo cualitativo, con el mismo consultor que en el paso 1 para poder comparar
    ejemplo_id = consultores["consultor_id"].iloc[0]
    ya_valorados = set(ratings_train.loc[ratings_train["consultor_id"] == ejemplo_id, "recurso_id"])
    candidatos = [r for r in todos_los_recursos if r not in ya_valorados]
    recomendaciones = recomendar_top_k(consultor_id=ejemplo_id, candidatos=candidatos, predecir_fn=predecir_user_based)
    log.info("Ejemplo (user-based) -> recomendaciones para consultor %d: %s", ejemplo_id, recomendaciones)


if __name__ == "__main__":
    main()
