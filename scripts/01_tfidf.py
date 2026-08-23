"""
Paso 1 de la demo: recomendador basado en contenido (TF-IDF).

Recomienda recursos comparando el perfil del consultor (tecnología principal +
sector actual) con la metadata de cada recurso (familia, tipo, tecnología,
sector), mediante similitud de texto. No necesita ningún historial de
valoraciones para funcionar -todo consultor tiene un perfil desde el primer
día-, por lo que resuelve el problema de cold-start que sí tienen los modelos
colaborativos de los siguientes pasos (CF, MF...).

Ejecución (desde recommenders/):
    uv run scripts/01_tfidf.py
"""

import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.config import K, RATING_RELEVANTE
from utils.data import cargar_datos, train_test_split_ratings
from utils.metrics import precision_at_k, hit_rate_at_k
from utils.logging_utils import get_logger

log = get_logger(nombre="01_tfidf")


def construir_matriz_tfidf(recursos: pd.DataFrame) -> tuple[TfidfVectorizer, spmatrix]:
    """Convierte la metadata de cada recurso en texto y ajusta un vectorizador TF-IDF."""
    texto = recursos["familia"] + " " + recursos["tipo"] + " " + recursos["tecnologia"] + " " + recursos["sector"]
    vectorizer = TfidfVectorizer()
    matriz = vectorizer.fit_transform(texto)
    return vectorizer, matriz


def recomendar_para_consultor(
    consultor_id: int,
    consultores: pd.DataFrame,
    ratings_train: pd.DataFrame,
    recursos: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matriz_tfidf: spmatrix,
    k: int = K,
) -> list[int]:
    """
    Recomienda recursos a un consultor comparando su perfil (tecnología + sector)
    con la metadata de cada recurso. Funciona igual con o sin historial de
    valoraciones: es la parte del cold-start real.
    """
    perfil = consultores.loc[consultores["consultor_id"] == consultor_id].iloc[0]
    consulta = f'{perfil["tecnologia_principal"]} {perfil["sector_actual"]}'
    vector_consultor = vectorizer.transform([consulta])

    similitudes = cosine_similarity(vector_consultor, matriz_tfidf).flatten()

    ya_valorados = set(ratings_train.loc[ratings_train["consultor_id"] == consultor_id, "recurso_id"])
    ranking = recursos.assign(similitud=similitudes).sort_values("similitud", ascending=False)
    ranking = ranking[~ranking["recurso_id"].isin(ya_valorados)]
    return ranking["recurso_id"].head(k).tolist()


def evaluar(
    consultores: pd.DataFrame,
    ratings_train: pd.DataFrame,
    ratings_test: pd.DataFrame,
    recursos: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matriz_tfidf: spmatrix,
    k: int = K,
) -> tuple[float, float]:
    """
    Precision@k y hit_rate@k medias: de los k recursos recomendados a cada
    consultor, qué proporción coincide con recursos que en el test valoró
    como relevantes (rating >= RATING_RELEVANTE), y si acertó al menos uno.
    """
    precisiones = []
    hit_rates = []
    for consultor_id in ratings_test["consultor_id"].unique():
        relevantes_test = ratings_test[
            (ratings_test["consultor_id"] == consultor_id) & (ratings_test["rating"] >= RATING_RELEVANTE)
        ]["recurso_id"].tolist()
        if not relevantes_test:
            continue

        recomendados = recomendar_para_consultor(
            consultor_id=consultor_id,
            consultores=consultores,
            ratings_train=ratings_train,
            recursos=recursos,
            vectorizer=vectorizer,
            matriz_tfidf=matriz_tfidf,
            k=k,
        )
        precisiones.append(precision_at_k(recomendados=recomendados, relevantes=relevantes_test, k=k))
        hit_rates.append(hit_rate_at_k(recomendados=recomendados, relevantes=relevantes_test, k=k))

    precision_media = sum(precisiones) / len(precisiones) if precisiones else 0.0
    hit_rate_media = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
    return precision_media, hit_rate_media


def main() -> None:
    consultores, recursos, ratings = cargar_datos()
    ratings_train, ratings_test = train_test_split_ratings(ratings=ratings)
    vectorizer, matriz_tfidf = construir_matriz_tfidf(recursos=recursos)

    log.info(
        "Consultores: %d | Recursos: %d | Ratings train/test: %d/%d",
        len(consultores), len(recursos), len(ratings_train), len(ratings_test),
    )

    precision_media, hit_rate_media = evaluar(
        consultores=consultores,
        ratings_train=ratings_train,
        ratings_test=ratings_test,
        recursos=recursos,
        vectorizer=vectorizer,
        matriz_tfidf=matriz_tfidf,
    )
    log.info(
        "TF-IDF content-based -> Precision@%d: %.3f | Hit Rate@%d: %.3f", K, precision_media, K, hit_rate_media
    )

    # Ejemplo cualitativo de cold-start: recomendaciones para un consultor
    # simulando que acaba de incorporarse (se ignora su historial real).
    ejemplo_id = consultores["consultor_id"].iloc[0]
    sin_historial = ratings_train.iloc[0:0]
    recomendaciones = recomendar_para_consultor(
        consultor_id=ejemplo_id,
        consultores=consultores,
        ratings_train=sin_historial,
        recursos=recursos,
        vectorizer=vectorizer,
        matriz_tfidf=matriz_tfidf,
    )
    log.info(
        "Ejemplo cold-start simulado (se ignora su historial a propósito) -> recomendaciones para consultor %d: %s",
        ejemplo_id, recomendaciones,
    )


if __name__ == "__main__":
    main()
