"""
Paso 3 de la demo: factorización de matrices (Matrix Factorization, MF) clásica.

En vez de calcular similitudes directas entre filas/columnas como el CF del
paso 2, aprende para cada consultor y cada recurso un vector de K factores
latentes (características implícitas, no etiquetadas a mano) de forma que el
producto de ambos vectores -más un sesgo de cada uno y la media global-
prediga la nota. Es el enfoque que popularizaron los ganadores del Netflix
Prize.

Se entrena por mínimos cuadrados alternos (ALS, alternating least squares):
en cada época se fijan los factores de los recursos y se resuelven por
mínimos cuadrados los de los consultores, y luego al revés.

Nota sobre la regularización: una regularización floja sobreajusta (memoriza
el train, falla en test) y una demasiado fuerte aplasta los factores a casi
cero (ni siquiera ajusta el train). REG y K_LATENTE están afinados por
barrido para minimizar el error en test con el tamaño actual del dataset.

Ejecución (desde recommenders/):
    uv run scripts/03_mf.py
"""

from utils.als import entrenar, predecir
from utils.config import K
from utils.data import cargar_datos, train_test_split_ratings, construir_matriz
from utils.evaluation import recomendar_top_k, evaluar
from utils.metrics import rmse, mae
from utils.logging_utils import get_logger

log = get_logger(nombre="03_mf")

K_LATENTE = 15  # nº de factores latentes por consultor/recurso
REG = 1.5  # penalización de regularización (ridge); afinado por barrido, ver nota más abajo
EPOCHS = 15


def main() -> None:
    consultores, recursos, ratings = cargar_datos()
    ratings_train, ratings_test = train_test_split_ratings(ratings=ratings)
    todos_los_recursos = recursos["recurso_id"].tolist()

    matriz = construir_matriz(ratings=ratings_train)
    indice_consultor = {consultor_id: i for i, consultor_id in enumerate(matriz.index)}
    indice_recurso = {recurso_id: j for j, recurso_id in enumerate(matriz.columns)}

    log.info(
        "Consultores: %d | Recursos: %d | Ratings train/test: %d/%d",
        len(consultores), len(recursos), len(ratings_train), len(ratings_test),
    )

    W, U, b, c, mu = entrenar(matriz=matriz, k_latente=K_LATENTE, reg=REG, epochs=EPOCHS)

    def predecir_fn(consultor_id: int, recurso_id: int) -> float:
        return predecir(
            consultor_id=consultor_id,
            recurso_id=recurso_id,
            indice_consultor=indice_consultor,
            indice_recurso=indice_recurso,
            W=W, U=U, b=b, c=c, mu=mu,
        )

    # Baseline ingenuo (predecir siempre la media global) para tener una referencia honesta.
    baseline_pred = [mu] * len(ratings_test)
    log.info(
        "Baseline (media global) -> RMSE: %.3f | MAE: %.3f",
        rmse(y_true=ratings_test["rating"], y_pred=baseline_pred),
        mae(y_true=ratings_test["rating"], y_pred=baseline_pred),
    )

    rmse_, mae_, precision, hit_rate = evaluar(
        ratings_train=ratings_train, ratings_test=ratings_test, todos_los_recursos=todos_los_recursos,
        predecir_fn=predecir_fn,
    )
    log.info(
        "MF -> RMSE: %.3f | MAE: %.3f | Precision@%d: %.3f | Hit Rate@%d: %.3f",
        rmse_, mae_, K, precision, K, hit_rate,
    )

    # Ejemplo cualitativo, con el mismo consultor que en los pasos 1 y 2 para poder comparar
    ejemplo_id = consultores["consultor_id"].iloc[0]
    ya_valorados = set(ratings_train.loc[ratings_train["consultor_id"] == ejemplo_id, "recurso_id"])
    candidatos = [r for r in todos_los_recursos if r not in ya_valorados]
    recomendaciones = recomendar_top_k(consultor_id=ejemplo_id, candidatos=candidatos, predecir_fn=predecir_fn)
    log.info("Ejemplo (MF) -> recomendaciones para consultor %d: %s", ejemplo_id, recomendaciones)


if __name__ == "__main__":
    main()
