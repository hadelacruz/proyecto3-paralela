"""Registro de uso de recursos durante cada fase (evidencia de ejecución).

``MonitorRecursos`` es un administrador de contexto que, mientras dura el bloque
``with``, guarda una muestra por intervalo con:

* el uso de **cada núcleo lógico** de la CPU y la RAM usada (``psutil``), y
* el uso de la GPU, de su memoria, la temperatura y la potencia (``nvidia-smi``).

Uso::

    with MonitorRecursos('resultados/monitor/preparacion_paralela.csv', fase='prep'):
        preparar_paralelo(...)
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import threading
import time

import psutil

CAMPOS_GPU = ['utilization.gpu', 'utilization.memory', 'memory.used', 'memory.total',
              'temperature.gpu', 'power.draw', 'clocks.sm']


def leer_gpu() -> dict:
    """Una lectura de nvidia-smi; vacía si no hay GPU NVIDIA disponible."""
    if shutil.which('nvidia-smi') is None:
        return {}
    try:
        salida = subprocess.run(
            ['nvidia-smi', f"--query-gpu={','.join(CAMPOS_GPU)}",
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout
        valores = [v.strip() for v in salida.strip().splitlines()[0].split(',')]
    except Exception:
        return {}
    return {f"gpu_{c.replace('.', '_')}": v for c, v in zip(CAMPOS_GPU, valores)}


class MonitorRecursos:
    def __init__(self, ruta_csv: str, fase: str = '', intervalo: float = 1.0,
                 con_gpu: bool = True):
        self.ruta_csv, self.fase, self.intervalo = ruta_csv, fase, intervalo
        self.con_gpu = con_gpu
        self.muestras: list[dict] = []
        self._detener = threading.Event()
        self._hilo = threading.Thread(target=self._bucle, daemon=True)

    def _bucle(self) -> None:
        inicio = time.perf_counter()
        psutil.cpu_percent(percpu=True)  # la primera lectura siempre es 0: se descarta
        while not self._detener.wait(self.intervalo):
            nucleos = psutil.cpu_percent(percpu=True)
            fila = {'fase': self.fase, 't': round(time.perf_counter() - inicio, 2),
                    'cpu_total': round(sum(nucleos) / len(nucleos), 1),
                    'ram_usada_gb': round(psutil.virtual_memory().used / 1e9, 2)}
            fila.update({f'cpu{i}': v for i, v in enumerate(nucleos)})
            if self.con_gpu:
                fila.update(leer_gpu())
            self.muestras.append(fila)

    def __enter__(self) -> 'MonitorRecursos':
        self._hilo.start()
        return self

    def __exit__(self, *exc) -> None:
        self._detener.set()
        self._hilo.join()
        if self.muestras:
            os.makedirs(os.path.dirname(self.ruta_csv) or '.', exist_ok=True)
            campos = list(dict.fromkeys(k for m in self.muestras for k in m))
            with open(self.ruta_csv, 'w', newline='', encoding='utf-8') as f:
                escritor = csv.DictWriter(f, fieldnames=campos)
                escritor.writeheader()
                escritor.writerows(self.muestras)

    def resumen(self) -> dict:
        """Promedios y picos del uso registrado."""
        if not self.muestras:
            return {}
        def numeros(clave):
            return [float(m[clave]) for m in self.muestras
                    if clave in m and str(m[clave]).replace('.', '', 1).isdigit()]
        res = {'cpu_total_media': sum(numeros('cpu_total')) / len(self.muestras)}
        gpu = numeros('gpu_utilization_gpu')
        if gpu:
            res['gpu_util_media'] = sum(gpu) / len(gpu)
            res['gpu_util_max'] = max(gpu)
            res['gpu_mem_max_mb'] = max(numeros('gpu_memory_used'))
        return res
