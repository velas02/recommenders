"""Logger sencillo compartido por los experimentos: escribe en consola y en un
fichero dentro de logs/<nombre>.log, para dejar constancia de cada ejecución
y de las métricas obtenidas.
"""

import logging
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"


def get_logger(nombre: str) -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger(nombre)
    if logger.handlers:  # evita duplicar handlers si se llama más de una vez
        return logger
    logger.setLevel(logging.INFO)

    formato = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    consola = logging.StreamHandler()
    consola.setFormatter(formato)
    logger.addHandler(consola)

    fichero = logging.FileHandler(LOGS_DIR / f"{nombre}.log")
    fichero.setFormatter(formato)
    logger.addHandler(fichero)

    return logger
