"""
Genera el dataset sintético para la demo del Knowledge Hub interno de Tinámica.

Al ejecutarlo crea tres ficheros CSV en esta misma carpeta:
  - consultores.csv : perfil de cada consultor (usuario)
  - recursos.csv    : catálogo de recursos de conocimiento (ítem)
  - ratings.csv     : valoraciones 1-5 que los consultores dan a los recursos que han usado

La generación no es puramente aleatoria: cada consultor tiene más probabilidad
de valorar -y valora mejor- los recursos afines a su tecnología o al sector de
su proyecto actual. Así el dataset tiene una estructura de preferencias real
que los modelos de recomendación pueden aprender (si fuera 100% aleatorio,
ningún modelo tendría patrones que encontrar).

Además de esa afinidad "de contenido" (tecnología, sector), hay una segunda
afinidad de "estilo": dos recursos pueden tratar el mismo tema y aun así
estar escritos de forma muy distinta (uno práctico, con ejemplos; otro
teórico y formal). Cada consultor también prefiere un estilo de trabajo, y
eso influye tanto en qué recursos usa como en la nota que les pone. Esta
preferencia de estilo se usa solo como paso intermedio del cálculo y no se
exporta a los CSV finales: no es una columna de metadata consultable, solo
se manifiesta en el patrón de valoraciones (ratings.csv).

Uso:
    python generate_dataset.py
"""

import random
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuración: aquí se ajustan el tamaño del dataset y los catálogos de valores
# ---------------------------------------------------------------------------

SEED = 42
N_CONSULTORES = 500
N_RECURSOS = 150
MIN_RATINGS_POR_CONSULTOR = 30
MAX_RATINGS_POR_CONSULTOR = 90
OUTPUT_DIR = Path(__file__).parent

ROLES = ["Data Scientist", "Ingeniero de Datos", "Consultor Funcional", "Arquitecto Cloud", "Consultor BI"]
SENIORITIES = ["Junior", "Senior", "Manager"]
TECNOLOGIAS = ["Python", "Spark", "AWS", "Azure", "SQL", "Power BI", "Java", "Kubernetes"]
SECTORES = ["Banca", "Seguros", "Retail", "Industria", "Utilities", "Telco"]
ESTILOS = ["Práctico", "Teórico", "Visual", "Conciso", "Detallado"]  # no depende de tecnología/sector/rol

TIPOS_TECNICOS = ["Plantilla", "Snippet", "Documentación"]
TIPOS_CASO_EXITO = ["Informe de proyecto", "Caso de éxito"]

NOMBRES = ["Ana", "Luis", "Marta", "Carlos", "Elena", "Javier", "Lucía", "Pablo", "Sara", "Diego",
           "Laura", "Miguel", "Paula", "Adrián", "Sofía", "Álvaro", "Nuria", "Hugo", "Marina", "Raúl"]
APELLIDOS = ["García", "Martínez", "López", "Sánchez", "Pérez", "Gómez", "Fernández", "Díaz",
             "Ruiz", "Moreno", "Álvarez", "Romero", "Alonso", "Gutiérrez", "Navarro", "Torres"]

random.seed(SEED)


# ---------------------------------------------------------------------------
# 1. Consultores (usuarios)
# ---------------------------------------------------------------------------

def generar_consultores(n: int) -> pd.DataFrame:
    consultores = []
    for i in range(1, n + 1):
        consultores.append({
            "consultor_id": i,
            "nombre": f"{random.choice(NOMBRES)} {random.choice(APELLIDOS)}",
            "rol": random.choice(ROLES),
            "seniority": random.choice(SENIORITIES),
            "tecnologia_principal": random.choice(TECNOLOGIAS),
            "sector_actual": random.choice(SECTORES),
            "estilo": random.choice(ESTILOS),
        })
    return pd.DataFrame(consultores)


# ---------------------------------------------------------------------------
# 2. Recursos (ítems)
# ---------------------------------------------------------------------------

def generar_recursos(n: int) -> pd.DataFrame:
    recursos = []
    for i in range(1, n + 1):
        familia = random.choice(["tecnico", "caso_exito"])
        tipo = random.choice(TIPOS_TECNICOS if familia == "tecnico" else TIPOS_CASO_EXITO)
        recursos.append({
            "recurso_id": i,
            "familia": familia,
            "tipo": tipo,
            "tecnologia": random.choice(TECNOLOGIAS),
            "sector": random.choice(SECTORES),
            "estilo": random.choice(ESTILOS),
        })
    return pd.DataFrame(recursos)


# ---------------------------------------------------------------------------
# 3. Ratings (interacciones)
# ---------------------------------------------------------------------------
# Cada consultor valora sólo una parte de los recursos (matriz dispersa, como
# en la realidad), y no elige qué recursos usar al azar: como en la vida real,
# tiene más probabilidad de usar recursos afines a su tecnología/sector que
# recursos sin relación con su trabajo. La nota depende de esa misma afinidad,
# más un poco de ruido para que no sea perfectamente predecible.

def peso_muestreo(afinidad: int) -> int:
    """Probabilidad relativa de que ese recurso 'le llegue' al consultor: se duplica por cada punto de afinidad."""
    return 2 ** afinidad


def nota_base(afinidad: int) -> int:
    """La afinidad (0-4) se traduce directamente en la nota de partida (1-5)."""
    return afinidad + 1


def calcular_afinidad(consultor: pd.Series, recurso: dict) -> int:
    """
    Afinidad combinada (0 a 4) entre un consultor y un recurso:
    - de contenido (0, 1 o 2): coincide tecnología y/o sector.
    - de estilo (0 o 2): coincide el estilo de trabajo/redacción preferido.
    """
    afinidad_contenido = 0
    if recurso["tecnologia"] == consultor["tecnologia_principal"]:
        afinidad_contenido += 1
    if recurso["sector"] == consultor["sector_actual"]:
        afinidad_contenido += 1

    afinidad_estilo = 2 if recurso["estilo"] == consultor["estilo"] else 0

    return afinidad_contenido + afinidad_estilo


def elegir_recursos_para_consultor(consultor: pd.Series, recursos_lista: list[dict], n: int) -> list[dict]:
    """Muestreo sin reemplazo ponderado por afinidad (en vez de puramente aleatorio)."""
    pool = list(recursos_lista)
    elegidos = []
    for _ in range(n):
        pesos = [peso_muestreo(calcular_afinidad(consultor, r)) for r in pool]
        elegido = random.choices(pool, weights=pesos, k=1)[0]
        pool.remove(elegido)
        elegidos.append(elegido)
    return elegidos


def generar_ratings(consultores: pd.DataFrame, recursos: pd.DataFrame) -> pd.DataFrame:
    recursos_lista = recursos.to_dict("records")

    ratings = []
    for _, consultor in consultores.iterrows():
        n_ratings = random.randint(MIN_RATINGS_POR_CONSULTOR, MAX_RATINGS_POR_CONSULTOR)
        recursos_valorados = elegir_recursos_para_consultor(consultor, recursos_lista, n_ratings)

        for recurso in recursos_valorados:
            base = nota_base(calcular_afinidad(consultor, recurso))
            rating = base + random.choices([-1, 0, 1], weights=[1, 4, 1])[0]
            rating = max(1, min(5, rating))

            ratings.append({
                "consultor_id": consultor["consultor_id"],
                "recurso_id": recurso["recurso_id"],
                "rating": rating,
            })
    return pd.DataFrame(ratings)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    consultores = generar_consultores(N_CONSULTORES)
    recursos = generar_recursos(N_RECURSOS)
    ratings = generar_ratings(consultores, recursos)  # usa "estilo" internamente, antes de descartarlo

    # "estilo" no se exporta: es una preferencia real pero que no está etiquetada en
    # ningún catálogo, solo se manifiesta en el patrón de valoraciones (ratings.csv).
    consultores.drop(columns="estilo").to_csv(OUTPUT_DIR / "consultores.csv", index=False)
    recursos.drop(columns="estilo").to_csv(OUTPUT_DIR / "recursos.csv", index=False)
    ratings.to_csv(OUTPUT_DIR / "ratings.csv", index=False)

    print(f"Consultores: {len(consultores):>4} -> consultores.csv")
    print(f"Recursos:    {len(recursos):>4} -> recursos.csv")
    print(f"Ratings:     {len(ratings):>4} -> ratings.csv")


if __name__ == "__main__":
    main()
