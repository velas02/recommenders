"""Parámetros compartidos por todos los pasos de la demo, para que los
resultados (precision@k, etc.) sean comparables entre un paso y el siguiente.
"""

K = 5  # nº de recomendaciones a generar por consultor
RATING_RELEVANTE = 4  # a partir de qué nota consideramos que un recurso "le gustó" al consultor
