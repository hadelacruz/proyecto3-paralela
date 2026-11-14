"""Fase 1 del pipeline: preparación de imágenes en la CPU.

Contiene dos implementaciones que producen exactamente el mismo tensor:

* ``preparar_secuencial``: el bucle del Ejercicio 3 del Lab 3 (un solo núcleo).
* ``preparar_paralelo``: la misma operación repartida entre varios procesos.

Por qué procesos y no hilos: decodificar un JPEG y redimensionarlo con Pillow es
trabajo de CPU que en buena parte se ejecuta bajo el GIL de Python, así que varios
hilos no escalan. Cada proceso tiene su propio intérprete y su propio GIL.

Por qué memoria compartida: si cada proceso devolviera su resultado como un arreglo
NumPy, ese resultado se serializaría (pickle) y se copiaría entre procesos: 1.14 GB
en float32 para 23,200 imágenes. En su lugar, el proceso principal reserva un solo
bloque de ``SharedMemory`` y cada proceso escribe directamente su tramo de filas.

Este archivo debe vivir en un módulo importable (no dentro del notebook): en Windows
los procesos se crean con ``spawn``, que reimporta las funciones por su nombre.
"""
from __future__ import annotations

import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory

import numpy as np
from PIL import Image

LIMITE_POR_CLASE = 800
NUEVA_RESOLUCION = (64, 64)
CANALES = 3
SEED = 42


def listar_clases(dataset_path: str) -> list[str]:
    """Clases en orden alfabético (A-Z, del, nothing, space), igual que el Lab 3."""
    return sorted(d for d in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, d)))


def seleccionar_archivos(dataset_path: str, clases: list[str],
                         limite: int = LIMITE_POR_CLASE,
                         seed: int = SEED) -> tuple[list[str], np.ndarray]:
    """Submuestreo reproducible de ``limite`` imágenes por clase.

    Corrige el problema de reproducibilidad del Lab 3: allí ``random.sample`` se
    llamaba antes de fijar la semilla y sobre ``os.listdir`` sin ordenar. Aquí se usa
    un generador propio con semilla fija y la lista de archivos ordenada, de modo que
    el subconjunto es idéntico en cada ejecución y en cada máquina.
    """
    rng = random.Random(seed)
    rutas, etiquetas = [], []
    for clase_idx, clase in enumerate(clases):
        carpeta = os.path.join(dataset_path, clase)
        archivos = sorted(os.listdir(carpeta))
        for archivo in rng.sample(archivos, min(limite, len(archivos))):
            rutas.append(os.path.join(carpeta, archivo))
            etiquetas.append(clase_idx)
    return rutas, np.asarray(etiquetas, dtype=np.int16)


def cargar_imagen(ruta: str) -> np.ndarray:
    """Lee, redimensiona y normaliza una imagen. Misma cadena que el Ejercicio 3."""
    img = Image.open(ruta).resize(NUEVA_RESOLUCION)
    return np.asarray(img, dtype=np.float32) / 255.0


def calentar_cache(rutas: list[str]) -> float:
    """Lee una vez los bytes de cada archivo para que queden en la caché del SO.

    El dataset vive en una carpeta de OneDrive, donde la primera lectura es mucho
    más lenta que las siguientes. Calentar la caché antes de medir hace que las
    versiones secuencial y paralela se comparen en igualdad de condiciones.
    Devuelve los segundos que tomó.
    """
    inicio = time.perf_counter()
    for ruta in rutas:
        with open(ruta, 'rb') as f:
            f.read()
    return time.perf_counter() - inicio


# --------------------------------------------------------------------------- #
# Versión secuencial (Lab 3, Ejercicio 3)
# --------------------------------------------------------------------------- #
def preparar_secuencial(rutas: list[str]) -> np.ndarray:
    """Procesa las imágenes una por una en el proceso actual."""
    X = np.empty((len(rutas), *NUEVA_RESOLUCION, CANALES), dtype=np.float32)
    for i, ruta in enumerate(rutas):
        X[i] = cargar_imagen(ruta)
    return X


# --------------------------------------------------------------------------- #
# Versión paralela (varios procesos + memoria compartida)
# --------------------------------------------------------------------------- #
def _trabajador(nombre_memoria: str, forma: tuple, inicio: int,
                rutas: list[str]) -> tuple[int, int, int]:
    """Procesa un tramo contiguo de imágenes y lo escribe en la memoria compartida.

    Devuelve (pid, inicio, cantidad) para poder registrar qué proceso hizo qué.
    """
    memoria = shared_memory.SharedMemory(name=nombre_memoria)
    try:
        X = np.ndarray(forma, dtype=np.float32, buffer=memoria.buf)
        for desplazamiento, ruta in enumerate(rutas):
            X[inicio + desplazamiento] = cargar_imagen(ruta)
        del X  # liberar la vista antes de cerrar el bloque
    finally:
        memoria.close()
    return os.getpid(), inicio, len(rutas)


def preparar_paralelo(rutas: list[str], procesos: int | None = None,
                      tramos_por_proceso: int = 4,
                      ejecutor: ProcessPoolExecutor | None = None
                      ) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Reparte las imágenes en tramos contiguos entre ``procesos`` procesos.

    Se crean ``procesos * tramos_por_proceso`` tramos (y no exactamente uno por
    proceso) para balancear la carga: si un proceso termina antes, toma otro tramo.

    Si se pasa un ``ejecutor`` ya creado se reutiliza; así se puede medir el tiempo
    de procesamiento sin incluir el arranque de los procesos (que en Windows,
    con ``spawn``, cuesta del orden de un segundo).

    Devuelve el tensor (copiado a memoria normal) y el registro de tramos.
    """
    procesos = procesos or os.cpu_count()
    forma = (len(rutas), *NUEVA_RESOLUCION, CANALES)
    nbytes = int(np.prod(forma)) * np.dtype(np.float32).itemsize
    memoria = shared_memory.SharedMemory(create=True, size=nbytes)
    propio = ejecutor is None
    ejecutor = ejecutor or ProcessPoolExecutor(max_workers=procesos)
    try:
        n_tramos = max(1, procesos * tramos_por_proceso)
        limites = np.linspace(0, len(rutas), n_tramos + 1, dtype=int)
        futuros = [ejecutor.submit(_trabajador, memoria.name, forma, int(a), rutas[a:b])
                   for a, b in zip(limites[:-1], limites[1:]) if b > a]
        registro = [f.result() for f in futuros]
        X = np.ndarray(forma, dtype=np.float32, buffer=memoria.buf).copy()
    finally:
        if propio:
            ejecutor.shutdown()
        memoria.close()
        memoria.unlink()
    return X, registro
