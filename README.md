# Proyecto 3: Mejorando algoritmos con CUDA

**CC3069 Computación Paralela y Distribuida, Universidad del Valle de Guatemala, semestre II-2026**

**Equipo:** Daniel Oswaldo Juárez Herrera · Humberto Alexander de la Cruz · Nicolle Alexandra Gordillo

Aceleración del **Laboratorio 3 de Data Science** (clasificación del alfabeto ASL con redes convolucionales). El pipeline se separa en dos fases, cada una en el hardware que mejor la ejecuta:

| Fase | Trabajo | Hardware | Implementación |
|---|---|---|---|
| 1. Preparación | leer, decodificar y redimensionar 23,200 JPEG | **CPU, 8 procesos** | `ProcessPoolExecutor` + `multiprocessing.shared_memory` |
| 2. Entrenamiento e inferencia | convoluciones, capas densas, retropropagación, Adam | **GPU (CUDA)** | PyTorch + cuDNN, y kernels propios en **Numba CUDA** |

## Estructura

```
proyecto3_cuda.ipynb      Entregable principal: pasos 1-6 del enunciado, ejecución en vivo y análisis
src/
  preparacion.py          Fase 1: versión secuencial (Lab 3) y paralela (procesos + memoria compartida)
  modelos_keras.py        Línea base: los modelos Keras/TensorFlow originales (CPU)
  modelos.py              Las mismas arquitecturas en PyTorch (equivalentes a Keras)
  entrenamiento.py        Bucle de entrenamiento con medición de transferencias y cómputo (eventos CUDA)
  kernels_cuda.py         CNN Base completa con kernels CUDA propios en Numba (adelante, atrás y Adam)
  monitor.py              Registro de uso por núcleo de CPU y de GPU (psutil + nvidia-smi)
  benchmark.py            Mediciones repetibles de cada fase
  correr_experimentos.py  Corre todos los benchmarks en secuencia
  analisis.py             Tablas, speedups y gráficas
  cifras_latex.py         Genera informe/cifras.tex (tablas del informe) desde resultados/
resultados/               CSV de tiempos, registros del monitor, logs y el Lab 3 ejecutado en esta máquina
evidencias/               Capturas de pantalla y salidas de nvidia-smi (ver GUIA_CAPTURAS.md)
informe/                  Informe.tex (LaTeX), cifras.tex generado y figuras/
lab3_original/            Notebook, informe y README del Lab 3 sin modificar (con su procedencia)
fotos_propias/            Fotos del equipo usadas en el Ejercicio 8 del Lab 3
```

## Cómo reproducir

### 1. Datos

El dataset **no se versiona** (~1.3 GB). Descarguen [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) de Kaggle y descompriman `asl_alphabet_train/` en la raíz del repositorio. También pueden indicar otra ubicación con la variable de entorno `ASL_DATASET_PATH`.

### 2. Entorno

Requisitos: Windows 11, Python 3.13, una GPU NVIDIA con un controlador que soporte CUDA 12.8 o superior.

Creen el entorno virtual **fuera de OneDrive**: si no, OneDrive intenta sincronizar los ~4 GB de PyTorch.

```powershell
python -m venv $env:USERPROFILE\venvs\proyecto3-cuda
& $env:USERPROFILE\venvs\proyecto3-cuda\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` fija las versiones exactas y documenta dos incompatibilidades que encontramos:
- El compilador de Numba tiene que ser la versión 12.8, igual que el `nvJitLink` que trae PyTorch.
- NumPy tiene que ser menor a 2.5, porque `numba-cuda` todavía usa `np.row_stack`.

### 3. Ejecutar

```powershell
# Todas las mediciones repetidas (~5 h en la laptop de prueba); resultados en resultados/
python -m src.correr_experimentos

# El notebook entregable (usa esos resultados y ejecuta en vivo cada fase, ~40 min)
python -m jupyter nbconvert --to notebook --execute --inplace proyecto3_cuda.ipynb --ExecutePreprocessor.timeout=-1

# Tablas del informe a partir de los resultados; luego compilar informe/Informe.tex con pdflatex
python -m src.cifras_latex
```

También se pueden correr pasos sueltos, por ejemplo `python -m src.benchmark preparacion --reps 5` o `python -m src.benchmark entrenamiento --framework torch --dispositivo cuda --modelo "CNN Base" --epocas 3 --reps 3`.

## Reproducibilidad del Lab 3

El Lab 3 original no era reproducible exactamente: tomaba la muestra de 800 imágenes por clase **antes** de fijar la semilla. El original se conserva intacto en `lab3_original/`. La corrección se aplica en:
- `src/preparacion.py` (`seleccionar_archivos`);
- `resultados/lab3_reproducible_ejecutado.ipynb`: el notebook del Lab 3 con solo esa corrección, ejecutado completo en esta máquina.

Los detalles, incluida una foto duplicada en `fotos_propias/`, están en `lab3_original/README.md`.

## Hardware de las mediciones

Intel Core i7-11390H (4 núcleos / 8 hilos, AVX-512) · 16 GB RAM · NVIDIA GeForce MX450 2 GB (Turing, modo WDDM) · Windows 11 · PyTorch 2.11 + CUDA 12.8 + cuDNN 9.19 · TensorFlow 2.21 (CPU) · Numba 0.68 + numba-cuda 0.30.
