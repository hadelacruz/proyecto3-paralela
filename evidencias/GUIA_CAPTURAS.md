# Guía de capturas de pantalla (evidencia de ejecución, 30 pts)

La rúbrica pide capturas de los monitores de recursos que demuestren que **cada fase usó el hardware que le correspondía**: los núcleos del CPU en la preparación de imágenes y la GPU en el entrenamiento.

El código ya registra esto automáticamente:
- `resultados/monitor/*.csv`: uso de cada núcleo y de la GPU, una muestra por segundo.
- `evidencias/nvidia-smi/*.txt`: salida completa de `nvidia-smi` cada 5 minutos.

Las capturas son el complemento visual que pide el enunciado. Guárdenlas en esta carpeta con los nombres de la tabla.

## Preparación (una sola vez)

1. **Administrador de Tareas → Rendimiento → CPU** → clic derecho en la gráfica → *Cambiar gráfico a* → **Procesadores lógicos**. Así se ven los 8 hilos por separado.
2. **Administrador de Tareas → Rendimiento → GPU 1 (NVIDIA GeForce MX450)**. En una de las cuatro gráficas pequeñas, cambien el menú desplegable a **Cuda** (por defecto muestra *3D*, que no refleja el cómputo CUDA).
3. Una terminal con `nvidia-smi -l 1` (se actualiza cada segundo).

## Capturas

| Archivo | Cuándo | Qué debe verse |
|---|---|---|
| `01_prep_secuencial_cpu.png` | Durante la celda **5.1 – preparación secuencial** del notebook | Un solo hilo lógico ocupado; el resto casi en reposo. |
| `02_prep_paralela_cpu.png` | Durante la celda **5.1 – preparación paralela (8 procesos)** | Los 8 hilos lógicos cerca del 100%. Opcional: pestaña *Detalles* con 8+ procesos `python.exe`. |
| `03_entrenamiento_gpu_taskmgr.png` | Durante la celda **5.2 – entrenamiento en GPU** | Gráfica *Cuda* de la MX450 alta y memoria dedicada de la GPU en uso. |
| `04_entrenamiento_gpu_nvidia_smi.png` | Igual que la anterior | `nvidia-smi` con `GPU-Util` alto, memoria usada y el proceso `python.exe` en la lista de procesos. |
| `05_entrenamiento_cpu_keras.png` | Durante una corrida de Keras en CPU (línea base) | Todos los hilos del CPU ocupados y la GPU en 0%. |
| `06_kernels_numba_gpu.png` | Durante la celda **5.3 – kernels Numba** | GPU ocupada por los kernels propios. |

## Consejos

- Laptop **conectada a la corriente** y sin otros programas pesados abiertos.
- Esperen unos 10 segundos después de que arranque la celda antes de capturar, para que la gráfica muestre el patrón y no el arranque.
- Si la frecuencia de la GPU baja durante el entrenamiento (se ve en `nvidia-smi` como `Pwr`/`Perf`), es el límite de potencia de la laptop y está documentado en el informe. No es un error.
