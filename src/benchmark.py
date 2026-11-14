"""Benchmarks del Proyecto 3: mismos datos y condiciones para CPU y GPU.

Fases medidas
-------------
1. Preparación (CPU): secuencial (1 proceso) contra paralela (1, 2, 4 y 8 procesos).
2. Entrenamiento: Keras/TF en CPU (original), PyTorch en CPU, PyTorch en GPU y
   CNN Base con kernels Numba CUDA propios. Número de épocas FIJO y sin
   EarlyStopping, para que todas las variantes hagan exactamente el mismo trabajo.
3. Inferencia sobre el conjunto de prueba (3,480 imágenes): CPU contra GPU.

Uso (desde la raíz del repo)::

    python -m src.benchmark preparacion --reps 5
    python -m src.benchmark entrenamiento --framework torch --dispositivo cuda \
        --modelo "CNN Base" --epocas 5 --reps 3

Cada corrida agrega filas a ``resultados/*.csv`` y guarda el registro del monitor
en ``resultados/monitor/``.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import sys
import time

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# P3_RESULTADOS permite redirigir la salida (p. ej. para pruebas rápidas)
RESULTADOS = os.environ.get('P3_RESULTADOS', os.path.join(RAIZ, 'resultados'))
SEED = 42

sys.path.insert(0, RAIZ)
from src import preparacion as prep  # noqa: E402
from src.monitor import MonitorRecursos  # noqa: E402


def ruta_dataset() -> str:
    return os.path.abspath(os.environ.get('ASL_DATASET_PATH',
                                          os.path.join(RAIZ, 'asl_alphabet_train')))


def agregar_csv(nombre: str, fila: dict) -> None:
    os.makedirs(RESULTADOS, exist_ok=True)
    ruta = os.path.join(RESULTADOS, nombre)
    nuevo = not os.path.exists(ruta)
    with open(ruta, 'a', newline='', encoding='utf-8') as f:
        escritor = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            escritor.writeheader()
        escritor.writerow(fila)


def marca() -> str:
    return dt.datetime.now().strftime('%Y%m%d_%H%M%S')


# --------------------------------------------------------------------------- #
# Datos: preparación + partición estratificada 70/15/15 (igual que el Lab 3)
# --------------------------------------------------------------------------- #
def cargar_datos(procesos: int | None = None):
    """Devuelve X_train, y_train, X_val, y_val, X_test, y_test, clases."""
    from sklearn.model_selection import train_test_split
    ds = ruta_dataset()
    clases = prep.listar_clases(ds)
    rutas, y = prep.seleccionar_archivos(ds, clases)
    X, _ = prep.preparar_paralelo(rutas, procesos)
    indices = np.arange(len(y))
    idx_temp, idx_test = train_test_split(indices, test_size=0.15, stratify=y,
                                          random_state=SEED)
    idx_train, idx_val = train_test_split(idx_temp, test_size=0.15 / 0.85,
                                          stratify=y[idx_temp], random_state=SEED)
    return (X[idx_train], y[idx_train], X[idx_val], y[idx_val],
            X[idx_test], y[idx_test], clases)


# --------------------------------------------------------------------------- #
# Fase 1: preparación
# --------------------------------------------------------------------------- #
def benchmark_preparacion(reps: int, procesos_lista=(1, 2, 4, 8)) -> None:
    from concurrent.futures import ProcessPoolExecutor
    ds = ruta_dataset()
    clases = prep.listar_clases(ds)
    rutas, _ = prep.seleccionar_archivos(ds, clases)
    print(f"{len(rutas):,} imágenes seleccionadas.")
    print(f"Calentando la caché del sistema de archivos... "
          f"{prep.calentar_cache(rutas):.1f}s")

    referencia = None
    for rep in range(1, reps + 1):
        with MonitorRecursos(os.path.join(RESULTADOS, 'monitor',
                                          f'prep_secuencial_rep{rep}.csv'),
                             fase='prep_secuencial', con_gpu=False):
            t0 = time.perf_counter()
            X_seq = prep.preparar_secuencial(rutas)
            t_seq = time.perf_counter() - t0
        if referencia is None:
            referencia = X_seq
        agregar_csv('preparacion.csv', {'rep': rep, 'variante': 'secuencial', 'procesos': 1,
                                        'segundos': round(t_seq, 3),
                                        'segundos_con_arranque': round(t_seq, 3),
                                        'identico': True})
        print(f"[rep {rep}] secuencial: {t_seq:.2f}s", flush=True)
        del X_seq

        for p in procesos_lista:
            t_arranque = time.perf_counter()
            with ProcessPoolExecutor(max_workers=p) as ejecutor:
                # Arranque: crear los procesos e importar el módulo en cada uno.
                list(ejecutor.map(_nada, range(p)))
                t_listo = time.perf_counter()
                with MonitorRecursos(os.path.join(RESULTADOS, 'monitor',
                                                  f'prep_paralela_{p}p_rep{rep}.csv'),
                                     fase=f'prep_paralela_{p}', con_gpu=False):
                    t0 = time.perf_counter()
                    X_par, _ = prep.preparar_paralelo(rutas, p, ejecutor=ejecutor)
                    t_par = time.perf_counter() - t0
            identico = bool(np.array_equal(X_par, referencia))
            agregar_csv('preparacion.csv', {
                'rep': rep, 'variante': 'paralela', 'procesos': p,
                'segundos': round(t_par, 3),
                'segundos_con_arranque': round(t_par + (t_listo - t_arranque), 3),
                'identico': identico})
            print(f"[rep {rep}] paralela {p} procesos: {t_par:.2f}s "
                  f"(+{t_listo - t_arranque:.2f}s de arranque) idéntico={identico}", flush=True)
            del X_par


def _nada(_):
    return os.getpid()


# --------------------------------------------------------------------------- #
# Fase 2 y 3: entrenamiento e inferencia
# --------------------------------------------------------------------------- #
def benchmark_entrenamiento(framework: str, dispositivo: str, modelo: str,
                            epocas: int, reps: int, lote: int = 64) -> None:
    datos = cargar_datos()
    X_train, y_train, X_val, y_val, X_test, y_test, _ = datos
    print(f"Datos: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    etiqueta = f"{framework}_{dispositivo}_{modelo.replace(' ', '').replace('+', '_')}"

    for rep in range(1, reps + 1):
        ruta_monitor = os.path.join(RESULTADOS, 'monitor', f'train_{etiqueta}_rep{rep}.csv')
        print(f"\n=== {framework} | {dispositivo} | {modelo} | rep {rep}/{reps} ===", flush=True)
        with MonitorRecursos(ruta_monitor, fase=f'train_{etiqueta}') as monitor:
            if framework == 'keras':
                res = _entrenar_keras(modelo, datos, epocas, lote)
            elif framework == 'torch':
                res = _entrenar_torch(modelo, dispositivo, datos, epocas, lote)
            elif framework == 'numba':
                res = _entrenar_numba(datos, epocas, lote)
            else:
                raise ValueError(framework)
        t_epocas = np.asarray(res['t_epoca'])
        fila = {
            'fecha': marca(), 'rep': rep, 'framework': framework, 'dispositivo': dispositivo,
            'modelo': modelo, 'epocas': epocas, 'lote': lote,
            't_total_s': round(float(t_epocas.sum()), 3),
            't_primera_epoca_s': round(float(t_epocas[0]), 3),
            # La primera época incluye compilación/autotuning; la media sin ella es
            # el costo en régimen estable.
            't_epoca_media_sin_1ra_s': round(float(t_epocas[1:].mean() if len(t_epocas) > 1
                                                    else t_epocas[0]), 3),
            't_transferencia_s': round(float(np.sum(res.get('t_transferencia', [0]))), 3),
            't_computo_gpu_s': round(float(np.sum(res.get('t_computo', [0]))), 3),
            'val_accuracy_final': round(float(res['val_accuracy'][-1]), 4),
            'test_accuracy': round(float(res['test_accuracy']), 4),
            't_inferencia_test_s': round(float(res['t_inferencia']), 4),
            'cpu_media_pct': round(monitor.resumen().get('cpu_total_media', float('nan')), 1),
            'gpu_util_media_pct': round(monitor.resumen().get('gpu_util_media', float('nan')), 1),
            'gpu_mem_max_mb': monitor.resumen().get('gpu_mem_max_mb', float('nan')),
        }
        agregar_csv('entrenamiento.csv', fila)
        with open(os.path.join(RESULTADOS, 'monitor', f'hist_{etiqueta}_rep{rep}.json'),
                  'w', encoding='utf-8') as f:
            json.dump({k: list(map(float, v)) if isinstance(v, (list, np.ndarray)) else v
                       for k, v in res.items()}, f, indent=1)
        print(json.dumps(fila, indent=1, ensure_ascii=False), flush=True)


def _entrenar_keras(modelo, datos, epocas, lote):
    from src.modelos_keras import entrenar_keras
    X_train, y_train, X_val, y_val, X_test, y_test, _ = datos
    m, hist, tiempos = entrenar_keras(modelo, X_train, y_train, X_val, y_val,
                                      epocas=epocas, lote=lote)
    m.predict(X_test[:256], batch_size=256, verbose=0)          # calentamiento
    t0 = time.perf_counter()
    pred = m.predict(X_test, batch_size=256, verbose=0).argmax(axis=1)
    t_inf = time.perf_counter() - t0
    return {'t_epoca': tiempos, 'val_accuracy': hist.history['val_accuracy'],
            'accuracy': hist.history['accuracy'], 'loss': hist.history['loss'],
            'val_loss': hist.history['val_loss'],
            'test_accuracy': (pred == y_test).mean(), 't_inferencia': t_inf}


def _entrenar_torch(modelo, dispositivo, datos, epocas, lote):
    import torch
    from src.entrenamiento import entrenar, evaluar, a_tensores
    from src.modelos import CONSTRUCTORES
    torch.backends.cudnn.benchmark = True
    if dispositivo == 'cuda':
        # En GPU la CPU solo arma lotes y encola kernels. Con los 4 hilos por
        # defecto, OpenMP deja los hilos en espera activa entre lote y lote y el
        # Administrador de Tareas muestra la CPU ocupada sin hacer trabajo útil.
        torch.set_num_threads(1)
    X_train, y_train, X_val, y_val, X_test, y_test, _ = datos
    m = CONSTRUCTORES[modelo]()
    hist = entrenar(m, X_train, y_train, X_val, y_val, dispositivo=dispositivo,
                    epocas=epocas, lote=lote)
    dev = torch.device(dispositivo)
    Xt, yt = a_tensores(X_test, y_test, pinned=dev.type == 'cuda')
    evaluar(m, Xt[:256], yt[:256], dev)                          # calentamiento
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _, acc, _ = evaluar(m, Xt, yt, dev)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    t_inf = time.perf_counter() - t0
    res = hist.como_dict()
    res.update({'test_accuracy': acc, 't_inferencia': t_inf})
    return res


def _entrenar_numba(datos, epocas, lote):
    from numba import cuda
    from src.kernels_cuda import entrenar_numba
    X_train, y_train, X_val, y_val, X_test, y_test, _ = datos
    m, hist = entrenar_numba(X_train, y_train, X_val, y_val, epocas=epocas, lote=lote)
    m.evaluar(X_test[:256], y_test[:256])                        # calentamiento
    cuda.synchronize()
    t0 = time.perf_counter()
    _, acc = m.evaluar(X_test, y_test)
    cuda.synchronize()
    hist.update({'test_accuracy': acc, 't_inferencia': time.perf_counter() - t0})
    return hist


def info_sistema() -> dict:
    import psutil
    info = {'python': platform.python_version(), 'so': platform.platform(),
            'cpu': platform.processor(), 'nucleos_fisicos': psutil.cpu_count(logical=False),
            'nucleos_logicos': psutil.cpu_count(), 'ram_gb': round(psutil.virtual_memory().total / 1e9, 1)}
    try:
        import torch
        info.update({'torch': torch.__version__, 'cuda_torch': torch.version.cuda,
                     'cudnn': torch.backends.cudnn.version(),
                     'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     'hilos_torch_cpu': torch.get_num_threads()})
    except ImportError:
        pass
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='fase', required=True)
    p1 = sub.add_parser('preparacion')
    p1.add_argument('--reps', type=int, default=5)
    p2 = sub.add_parser('entrenamiento')
    p2.add_argument('--framework', choices=['keras', 'torch', 'numba'], required=True)
    p2.add_argument('--dispositivo', choices=['cpu', 'cuda'], default='cuda')
    p2.add_argument('--modelo', default='CNN Base')
    p2.add_argument('--epocas', type=int, default=5)
    p2.add_argument('--reps', type=int, default=3)
    p2.add_argument('--lote', type=int, default=64)
    sub.add_parser('sistema')
    args = ap.parse_args()

    if args.fase == 'preparacion':
        benchmark_preparacion(args.reps)
    elif args.fase == 'entrenamiento':
        benchmark_entrenamiento(args.framework, args.dispositivo, args.modelo,
                                args.epocas, args.reps, args.lote)
    else:
        print(json.dumps(info_sistema(), indent=1))


if __name__ == '__main__':
    main()
