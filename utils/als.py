"""
ALS (mínimos cuadrados alternos) para factorización de matrices clásica.

Reutilizado por el MF del paso 3 y, además, como "calentamiento" del MF con
Keras del paso 4: en vez de arrancar sus embeddings desde pesos aleatorios,
se inicializan con esta misma solución ya entrenada.
"""

import numpy as np
import pandas as pd


def entrenar(
    matriz: pd.DataFrame, k_latente: int, reg: float, epochs: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Devuelve W (factores de consultor), U (factores de recurso), b (sesgo de
    consultor), c (sesgo de recurso) y mu (media global), tras `epochs` rondas
    de ALS. prediccion[i, j] = W[i] . U[j] + b[i] + c[j] + mu
    """
    R = matriz.to_numpy()
    valorado = ~np.isnan(R)
    n_consultores, n_recursos = R.shape
    mu = np.nanmean(R)

    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_consultores, k_latente))
    U = rng.normal(size=(n_recursos, k_latente))
    b = np.zeros(n_consultores)
    c = np.zeros(n_recursos)

    identidad = np.eye(k_latente) * reg

    for _ in range(epochs):
        # fija U y c, resuelve W y b consultor a consultor
        for i in range(n_consultores):
            vistos = valorado[i]
            if not vistos.any():
                continue
            U_vistos = U[vistos]
            r = R[i, vistos]
            matriz_sistema = U_vistos.T @ U_vistos + identidad  # (K,m)@(m,K) -> (K,K), con m = recursos vistos ⊆ M
            vector_sistema = U_vistos.T @ (r - b[i] - c[vistos] - mu)  # (K,m)@(m,) -> (K,)
            bi = (r - U_vistos @ W[i] - c[vistos] - mu).sum()  # suma de errores (nota real - predicción sin sesgo b[i], con W[i] aún viejo)

            W[i] = np.linalg.solve(matriz_sistema, vector_sistema)  # Ax=b: A=U_vistosᵀU_vistos+λI, b=U_vistosᵀ·residuo, x=W[i] -> ecuación normal de la regresión ridge
            b[i] = bi / (vistos.sum() + reg)  # media (regularizada) del error que no explican los factores: el sesgo propio del consultor

        # fija W y b, resuelve U y c recurso a recurso
        for j in range(n_recursos):
            vistos = valorado[:, j]
            if not vistos.any():
                continue
            W_vistos = W[vistos]
            r = R[vistos, j]
            matriz_sistema = W_vistos.T @ W_vistos + identidad  # (K,n)@(n,K) -> (K,K), con n = consultores vistos ⊆ N
            vector_sistema = W_vistos.T @ (r - b[vistos] - c[j] - mu)  # (K,n)@(n,) -> (K,)
            cj = (r - W_vistos @ U[j] - b[vistos] - mu).sum()  # suma de errores (nota real - predicción sin sesgo c[j], con U[j] aún viejo)

            U[j] = np.linalg.solve(matriz_sistema, vector_sistema)  # Ax=b: A=W_vistosᵀW_vistos+λI, b=W_vistosᵀ·residuo, x=U[j] -> ecuación normal de la regresión ridge
            c[j] = cj / (vistos.sum() + reg)  # media (regularizada) del error que no explican los factores: el sesgo propio del recurso

    return W, U, b, c, mu


def predecir(
    consultor_id: int, recurso_id: int, indice_consultor: dict[int, int], indice_recurso: dict[int, int],
    W: np.ndarray, U: np.ndarray, b: np.ndarray, c: np.ndarray, mu: float,
) -> float:
    """Predice la nota como el producto de los factores latentes, más los sesgos y la media global."""
    if consultor_id not in indice_consultor or recurso_id not in indice_recurso:
        return mu

    i = indice_consultor[consultor_id]
    j = indice_recurso[recurso_id]
    prediccion = W[i] @ U[j] + b[i] + c[j] + mu
    return min(5, max(1, prediccion))
