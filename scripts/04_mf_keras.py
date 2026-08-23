"""
Paso 4 de la demo: factorización de matrices con Keras (deep learning).

Sobre la misma idea de factores latentes del paso 3, pero en vez de un
producto escalar resuelto por mínimos cuadrados (ALS), una red neuronal
aprende a combinar los embeddings de consultor y recurso: se concatenan y se
pasan por una capa densa, que aprende una combinación no lineal entre ambos
(el producto escalar de MF es solo un caso particular de combinación, entre
muchas posibles).

Nota sobre las formas (shapes): cada Embedding recibe un id por muestra y
devuelve forma (batch, 1, k_latente) -el "1" es ese único id-. Flatten() lo
deja en (batch, k_latente) para poder concatenar los dos embeddings en un
único vector (batch, 2*k_latente) antes de pasarlo por la capa densa.

Nota sobre los hiperparámetros: entrenar los embeddings desde pesos
aleatorios con descenso de gradiente (Adam) necesita bastante más margen que
el ALS del paso 3 para llegar a superarlo -no unas pocas épocas, sino
cientos-, y regularización (Dropout) para que la capa densa no memorice el
train antes de aprender algo útil. Por eso `EPOCHS`/`PACIENCIA` son
deliberadamente altos.

Nota sobre el entrenamiento: el modelo se entrena para predecir la
desviación respecto a la media (`rating - mu`), no la nota en bruto -suele
ser más fácil de optimizar para la red que números centrados lejos de cero-,
así que hay que sumarle `mu` de vuelta a la salida para obtener la nota real
(se ve en `predecir_matriz_completa`).

Ejecución (desde recommenders/):
    uv run scripts/04_mf_keras.py
"""

import numpy as np
import pandas as pd
from tensorflow import keras

from utils.config import K
from utils.data import cargar_datos, train_test_split_ratings, indexar_ids
from utils.evaluation import recomendar_top_k, evaluar
from utils.metrics import rmse, mae
from utils.logging_utils import get_logger

keras.utils.set_random_seed(42)  # reproducibilidad: mismos pesos iniciales y mismo orden de batches

log = get_logger(nombre="04_mf_keras")

K_LATENTE = 20  # tamaño del embedding de consultor/recurso
UNIDADES_OCULTAS = 256  # tamaño de la capa densa
DROPOUT = 0.2  # regularización -sin esto sobreajusta antes de aprender nada útil-
LEARNING_RATE = 0.002
BATCH_SIZE = 256
EPOCHS = 2000  # tope de seguridad; en la práctica para antes por la parada temprana
PACIENCIA = 150  # épocas sin mejorar el val_loss antes de parar y quedarse con la mejor


def construir_modelo(n_consultores: int, n_recursos: int, k_latente: int = K_LATENTE) -> keras.Model:
    entrada_consultor = keras.Input(shape=(1,))
    entrada_recurso = keras.Input(shape=(1,))
    emb_consultor = keras.layers.Embedding(n_consultores, k_latente)(entrada_consultor)  # (batch, 1, k_latente)
    emb_recurso = keras.layers.Embedding(n_recursos, k_latente)(entrada_recurso)  # (batch, 1, k_latente)

    x = keras.layers.Concatenate()([keras.layers.Flatten()(emb_consultor), keras.layers.Flatten()(emb_recurso)])
    # x: (batch, 2*k_latente) -> los dos vectores concatenados uno detrás del otro
    x = keras.layers.Dropout(DROPOUT)(x)
    x = keras.layers.Dense(UNIDADES_OCULTAS, activation="relu")(x)
    salida = keras.layers.Dense(1)(x)  # (batch, 1): una nota (respecto a la media) por par consultor-recurso

    modelo = keras.Model(inputs=[entrada_consultor, entrada_recurso], outputs=salida)
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
    return modelo


def arrays_de_entrada(
    ratings: pd.DataFrame, indice_consultor: dict[int, int], indice_recurso: dict[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Convierte los consultor_id/recurso_id de un DataFrame de ratings en índices 0..n-1."""
    x_consultor = ratings["consultor_id"].map(indice_consultor).to_numpy()
    x_recurso = ratings["recurso_id"].map(indice_recurso).to_numpy()
    return x_consultor, x_recurso


def predecir_matriz_completa(modelo: keras.Model, n_consultores: int, n_recursos: int, mu: float) -> np.ndarray:
    """
    Predice la nota para TODOS los pares consultor-recurso de una sola vez
    (una única llamada al modelo, en vez de una por par) y la deja en una
    matriz [n_consultores, n_recursos] -mucho más rápido: Keras tiene un
    coste fijo por llamada que penaliza mucho si se predice par a par.
    """
    x_consultor = np.repeat(np.arange(n_consultores), n_recursos)
    x_recurso = np.tile(np.arange(n_recursos), n_consultores)
    predicciones = modelo.predict([x_consultor, x_recurso], batch_size=8192, verbose=0).flatten() + mu
    return predicciones.reshape(n_consultores, n_recursos)  # fila=consultor, columna=recurso


def main() -> None:
    consultores, recursos, ratings = cargar_datos()
    ratings_train, ratings_test = train_test_split_ratings(ratings=ratings)
    todos_los_recursos = recursos["recurso_id"].tolist()

    indice_consultor = indexar_ids(consultores["consultor_id"])
    indice_recurso = indexar_ids(recursos["recurso_id"])
    n_consultores, n_recursos = len(indice_consultor), len(indice_recurso)

    x_consultor_train, x_recurso_train = arrays_de_entrada(ratings_train, indice_consultor, indice_recurso)
    x_consultor_test, x_recurso_test = arrays_de_entrada(ratings_test, indice_consultor, indice_recurso)
    mu = ratings_train["rating"].mean()
    y_train = ratings_train["rating"].to_numpy(dtype=float) - mu
    y_test = ratings_test["rating"].to_numpy(dtype=float) - mu

    log.info(
        "Consultores: %d | Recursos: %d | Ratings train/test: %d/%d",
        len(consultores), len(recursos), len(ratings_train), len(ratings_test),
    )

    baseline_pred = [mu] * len(ratings_test)
    log.info(
        "Baseline (media global) -> RMSE: %.3f | MAE: %.3f",
        rmse(y_true=ratings_test["rating"], y_pred=baseline_pred),
        mae(y_true=ratings_test["rating"], y_pred=baseline_pred),
    )

    modelo = construir_modelo(n_consultores, n_recursos)
    parada_temprana = keras.callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA, restore_best_weights=True)
    modelo.fit(
        x=[x_consultor_train, x_recurso_train],
        y=y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=([x_consultor_test, x_recurso_test], y_test),
        callbacks=[parada_temprana],
        verbose=0,
    )

    matriz_predicciones = predecir_matriz_completa(modelo, n_consultores, n_recursos, mu)

    def predecir_fn(consultor_id: int, recurso_id: int) -> float:
        prediccion = matriz_predicciones[indice_consultor[consultor_id], indice_recurso[recurso_id]]
        return min(5, max(1, prediccion))

    rmse_, mae_, precision, hit_rate = evaluar(
        ratings_train=ratings_train, ratings_test=ratings_test, todos_los_recursos=todos_los_recursos,
        predecir_fn=predecir_fn,
    )
    log.info(
        "MF Keras (deep) -> RMSE: %.3f | MAE: %.3f | Precision@%d: %.3f | Hit Rate@%d: %.3f",
        rmse_, mae_, K, precision, K, hit_rate,
    )

    # Ejemplo cualitativo, mismo consultor que en pasos anteriores
    ejemplo_id = consultores["consultor_id"].iloc[0]
    ya_valorados = set(ratings_train.loc[ratings_train["consultor_id"] == ejemplo_id, "recurso_id"])
    candidatos = [r for r in todos_los_recursos if r not in ya_valorados]
    recomendaciones = recomendar_top_k(consultor_id=ejemplo_id, candidatos=candidatos, predecir_fn=predecir_fn)
    log.info("Ejemplo (MF Keras) -> recomendaciones para consultor %d: %s", ejemplo_id, recomendaciones)


if __name__ == "__main__":
    main()
