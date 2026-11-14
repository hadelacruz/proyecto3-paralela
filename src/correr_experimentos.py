"""Ejecuta todos los experimentos del Proyecto 3 en secuencia.

Se corren uno tras otro (nunca en paralelo) para que ninguna medición compita por
CPU, GPU o disco con otra. Cada paso escribe su salida en ``resultados/logs/``.

Uso (desde la raíz del repo, con el entorno virtual activo)::

    python -m src.correr_experimentos            # todo (~5 h en la laptop de prueba)
    python -m src.correr_experimentos lab3 prep  # solo algunos pasos

Pasos disponibles: prep, keras_cpu, torch_gpu, numba_gpu, torch_cpu, lab3.

Recomendaciones antes de correr: laptop conectada a la corriente, plan de energía
de alto rendimiento y ningún otro programa pesado abierto.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(RAIZ, 'resultados', 'logs')
PY = sys.executable

MODELOS = ['CNN Base', 'CNN Profunda', 'CNN Base + Aug', 'CNN Profunda + Aug']
EPOCAS, REPS = 3, 3
# Pausa entre corridas: CPU y GPU de una laptop bajan su frecuencia al calentarse
# (la MX450 llega al límite de potencia, "SW Power Cap", en ~30 s de carga). Esperar
# entre corridas hace que cada una arranque desde un estado térmico parecido.
PAUSA_S = 60


def bench(*args):
    return [PY, '-m', 'src.benchmark', *args]


def entrenamiento(framework, dispositivo, modelo, epocas=EPOCAS, reps=REPS):
    return bench('entrenamiento', '--framework', framework, '--dispositivo', dispositivo,
                 '--modelo', modelo, '--epocas', str(epocas), '--reps', str(reps))


PASOS = {
    'prep': [bench('preparacion', '--reps', '5')],
    'keras_cpu': [entrenamiento('keras', 'cpu', m) for m in MODELOS],
    'torch_gpu': [entrenamiento('torch', 'cuda', m) for m in MODELOS],
    'numba_gpu': [entrenamiento('numba', 'cuda', 'CNN Base')],
    # PyTorch en CPU es varias veces más lento que TF en CPU: solo se corre para aislar
    # el efecto del dispositivo (mismo código, distinto hardware), con menos épocas.
    'torch_cpu': [entrenamiento('torch', 'cpu', 'CNN Base', epocas=3, reps=1),
                  entrenamiento('torch', 'cpu', 'CNN Profunda', epocas=2, reps=1)],
    'lab3': ['lab3'],
}


def preparar_lab3_reproducible() -> str:
    """Copia del notebook original con la semilla corregida, lista para ejecutar.

    Único cambio: en la celda del submuestreo se fija la semilla y se ordena la
    lista de archivos ANTES de ``random.sample``. Todo lo demás queda intacto.
    """
    import nbformat
    origen = os.path.join(RAIZ, 'lab3_original', 'Lab3_Final.ipynb')
    nb = nbformat.read(origen, as_version=4)
    viejo = ("    archivos = os.listdir(ruta_clase)\n"
             "    archivos_por_clase[clase] = random.sample(")
    nuevo = ("    archivos = sorted(os.listdir(ruta_clase))  # [Proyecto 3] orden estable\n"
             "    archivos_por_clase[clase] = random.sample(")
    parchadas = 0
    for celda in nb.cells:
        if celda.cell_type == 'code' and viejo in celda.source:
            celda.source = ("# [Proyecto 3] La semilla se fija ANTES del submuestreo para que el\n"
                            "# subconjunto de 23,200 imágenes sea el mismo en cada ejecución.\n"
                            "random.seed(42)\n" + celda.source.replace(viejo, nuevo))
            parchadas += 1
    assert parchadas == 1, 'No se encontró la celda de submuestreo del Lab 3'
    destino = os.path.join(RAIZ, 'resultados', 'lab3_reproducible.ipynb')
    nbformat.write(nb, destino)
    return destino


def correr(paso: str, comando) -> dict:
    os.makedirs(LOGS, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
               ASL_DATASET_PATH=os.path.join(RAIZ, 'asl_alphabet_train'),
               FOTOS_PROPIAS_PATH=os.path.join(RAIZ, 'fotos_propias'))
    if comando == 'lab3':
        nb = preparar_lab3_reproducible()
        comando = [PY, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
                   '--ExecutePreprocessor.timeout=-1', '--output',
                   'lab3_reproducible_ejecutado.ipynb', nb]
        nombre = 'lab3'
    else:
        nombre = '_'.join(a.replace(' ', '').replace('+', '_') for a in comando[3:]
                          if not a.startswith('--'))[:80]
    ruta_log = os.path.join(LOGS, f'{paso}__{nombre}.log')
    print(f"[{time.strftime('%H:%M:%S')}] {paso}: {' '.join(comando[1:])}", flush=True)
    inicio = time.perf_counter()
    with open(ruta_log, 'w', encoding='utf-8') as log:
        codigo = subprocess.run(comando, cwd=RAIZ, env=env, stdout=log,
                                stderr=subprocess.STDOUT).returncode
    duracion = time.perf_counter() - inicio
    print(f"    -> código {codigo}, {duracion / 60:.1f} min, log: {os.path.relpath(ruta_log, RAIZ)}",
          flush=True)
    return {'paso': paso, 'log': os.path.relpath(ruta_log, RAIZ), 'codigo': codigo,
            'minutos': round(duracion / 60, 2)}


def main():
    pedidos = sys.argv[1:] or list(PASOS)
    resumen = []
    for paso in pedidos:
        for comando in PASOS[paso]:
            if resumen:
                time.sleep(PAUSA_S)
            resumen.append(correr(paso, comando))
    with open(os.path.join(LOGS, f"resumen_{time.strftime('%Y%m%d_%H%M%S')}.json"), 'w',
              encoding='utf-8') as f:
        json.dump(resumen, f, indent=1, ensure_ascii=False)
    fallidos = [r for r in resumen if r['codigo'] != 0]
    print(f"\nTerminado: {len(resumen) - len(fallidos)} pasos correctos, {len(fallidos)} con error.")


if __name__ == '__main__':
    main()
