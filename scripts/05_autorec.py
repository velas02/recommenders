"""
Paso 5 de la demo: AutoRec (autoencoder), deep learning avanzado.

En vez de aprender un vector latente por consultor y otro por recurso como en
los pasos 3 y 4, un autoencoder aprende a reconstruir directamente la fila
completa de notas de un consultor (una nota por cada uno de los N recursos):
la comprime en una capa oculta más pequeña -el cuello de botella, que aquí
hace de "factores latentes" aprendidos implícitamente- y la reconstruye a la
salida. Al no poder copiar la entrada tal cual, el modelo se ve obligado a
aprender los patrones que explican las notas en vez de memorizarlas.

Nota sobre las notas que faltan: la mayoría de celdas consultor-recurso no
tienen nota (dataset disperso). Se rellenan a 0 para poder usar una matriz
densa como entrada, y una máscara (1 si hay nota, 0 si no) le dice a la
función de pérdida qué celdas ignorar -si no, el modelo aprendería a predecir
0 para casi todo, justo lo contrario de lo que queremos-.

Nota sobre validación: el modelo siempre recibe como entrada la fila de
train (con ceros donde falta nota). Para medir si generaliza, el objetivo de
validación no repite esa misma fila, sino las notas que se dejaron aparte en
test -así el val_loss mide reconstrucción sobre celdas nunca vistas en train-.

Ejecución (desde recommenders/):
    uv run scripts/05_autorec.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from utils.config import K
from utils.data import cargar_datos, train_test_split_ratings, indexar_ids
from utils.evaluation import recomendar_top_k, evaluar
from utils.metrics import rmse, mae
from utils.logging_utils import get_logger

keras.utils.set_random_seed(42)  # reproducibilidad: mismos pesos iniciales y mismo orden de batches

log = get_logger(nombre="05_autorec")

UNIDADES_OCULTAS = 64  # tamaño del cuello de botella (factores latentes implícitos)
DROPOUT = 0.3  # regularización sobre la fila de entrada
REG = 1e-4  # regularización L2 de las capas densas
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 2000  # tope de seguridad; en la práctica para antes por la parada temprana
PACIENCIA = 150  # épocas sin mejorar el val_loss antes de parar y quedarse con la mejor


def construir_perdida_enmascarada(n_recursos: int):
    """
    MSE que solo cuenta las celdas con nota real. `y_true` llega con las
    notas y la máscara concatenadas en columnas (ver `objetivo_con_mascara`)
    para que ambas viajen siempre juntas -incluso si Keras baraja las filas
    entre épocas, notas y máscara nunca se desalinean entre sí-.
    """

    def perdida(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        notas, mascara = y_true[:, :n_recursos], y_true[:, n_recursos:]
        diff = (y_pred - notas) * mascara
        return tf.reduce_sum(tf.square(diff)) / (tf.reduce_sum(mascara) + 1e-8)

    return perdida


def construir_modelo(n_recursos: int) -> keras.Model:
    entrada = keras.Input(shape=(n_recursos,))
    x = keras.layers.Dropout(DROPOUT)(entrada)
    x = keras.layers.Dense(UNIDADES_OCULTAS, activation="tanh", kernel_regularizer=keras.regularizers.l2(REG))(x)
    salida = keras.layers.Dense(n_recursos, kernel_regularizer=keras.regularizers.l2(REG))(x)  # reconstrucción

    modelo = keras.Model(inputs=entrada, outputs=salida)
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=construir_perdida_enmascarada(n_recursos))
    return modelo


def matriz_y_mascara(
    ratings_df: pd.DataFrame, indice_consultor: dict[int, int], indice_recurso: dict[int, int],
    n_consultores: int, n_recursos: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Notas en bruto y máscara (1 si hay nota) como matrices [n_consultores, n_recursos]."""
    notas = np.zeros((n_consultores, n_recursos))
    mascara = np.zeros((n_consultores, n_recursos))
    filas = ratings_df["consultor_id"].map(indice_consultor).to_numpy()
    columnas = ratings_df["recurso_id"].map(indice_recurso).to_numpy()
    notas[filas, columnas] = ratings_df["rating"].to_numpy(dtype=float)
    mascara[filas, columnas] = 1.0
    return notas, mascara


def objetivo_con_mascara(notas_centradas: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Concatena notas y máscara en un único array para pasarlo como `y` a fit/validation_data."""
    return np.concatenate([notas_centradas, mascara], axis=1)


def main() -> None:
    consultores, recursos, ratings = cargar_datos()
    ratings_train, ratings_test = train_test_split_ratings(ratings=ratings)
    todos_los_recursos = recursos["recurso_id"].tolist()

    indice_consultor = indexar_ids(consultores["consultor_id"])
    indice_recurso = indexar_ids(recursos["recurso_id"])
    n_consultores, n_recursos = len(indice_consultor), len(indice_recurso)

    log.info(
        "Consultores: %d | Recursos: %d | Ratings train/test: %d/%d",
        len(consultores), len(recursos), len(ratings_train), len(ratings_test),
    )

    notas_train, mascara_train = matriz_y_mascara(ratings_train, indice_consultor, indice_recurso, n_consultores, n_recursos)
    notas_test, mascara_test = matriz_y_mascara(ratings_test, indice_consultor, indice_recurso, n_consultores, n_recursos)

    mu = ratings_train["rating"].mean()
    entrada = (notas_train - mu) * mascara_train  # centrado; las celdas sin nota se quedan a 0
    objetivo_train = objetivo_con_mascara(entrada, mascara_train)
    objetivo_test = objetivo_con_mascara((notas_test - mu) * mascara_test, mascara_test)

    baseline_pred = [mu] * len(ratings_test)
    log.info(
        "Baseline (media global) -> RMSE: %.3f | MAE: %.3f",
        rmse(y_true=ratings_test["rating"], y_pred=baseline_pred),
        mae(y_true=ratings_test["rating"], y_pred=baseline_pred),
    )

    modelo = construir_modelo(n_recursos)
    parada_temprana = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA, restore_best_weights=True)
    modelo.fit(
        x=entrada,
        y=objetivo_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(entrada, objetivo_test),
        callbacks=[parada_temprana],
        verbose=0,
    )

    matriz_predicciones = modelo.predict(entrada, batch_size=n_consultores, verbose=0) + mu

    def predecir_fn(consultor_id: int, recurso_id: int) -> float:
        prediccion = matriz_predicciones[indice_consultor[consultor_id], indice_recurso[recurso_id]]
        return min(5, max(1, prediccion))

    rmse_, mae_, precision, hit_rate = evaluar(
        ratings_train=ratings_train, ratings_test=ratings_test, todos_los_recursos=todos_los_recursos,
        predecir_fn=predecir_fn,
    )
    log.info(
        "AutoRec -> RMSE: %.3f | MAE: %.3f | Precision@%d: %.3f | Hit Rate@%d: %.3f",
        rmse_, mae_, K, precision, K, hit_rate,
    )

    # Ejemplo cualitativo, mismo consultor que en pasos anteriores
    ejemplo_id = consultores["consultor_id"].iloc[0]
    ya_valorados = set(ratings_train.loc[ratings_train["consultor_id"] == ejemplo_id, "recurso_id"])
    candidatos = [r for r in todos_los_recursos if r not in ya_valorados]
    recomendaciones = recomendar_top_k(consultor_id=ejemplo_id, candidatos=candidatos, predecir_fn=predecir_fn)
    log.info("Ejemplo (AutoRec) -> recomendaciones para consultor %d: %s", ejemplo_id, recomendaciones)


if __name__ == "__main__":
    main()
