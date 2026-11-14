# Mediciones descartadas

`entrenamiento_keras_cpu_ram_saturada.csv` y `monitor/` son la primera medición de la línea base (Keras/TF en CPU, 13-nov-2026, 16:54-18:27). Se descartaron porque la RAM estuvo entre 14 y 15.8 GB de 16 durante toda la corrida, y al final el sistema se quedó sin memoria y detuvo el proceso.

Esas épocas tardaron 1.8 veces más que las del Lab 3 completo, que corrió en la misma laptop esa misma noche con memoria libre:

| Modelo | Esta medición (s/época) | Lab 3 completo (s/época) |
|---|---|---|
| CNN Base | 52 | 29 |
| CNN Profunda | 289 | 147 |
| CNN Base + Aug | 57 | 36 |
| CNN Profunda + Aug | 273 | 155 |

Si se usaran, el speedup de la GPU saldría inflado casi al doble. La línea base se volvió a medir con la memoria libre, y el resultado está en `resultados/entrenamiento.csv`.

## Segundo intento (13-nov-2026, 23:41-23:56)

`entrenamiento_keras_cpu_modo_eficiencia.csv`: Windows tenía activo el modo de energía "Mejor eficiencia energética" (overlay `961cc777-…`). El CPU estaba ocupado al 90-100 %, pero corría a ~27 % de su frecuencia máxima, y las épocas de la CNN Base tardaron entre 51 y 116 s. Se cambió a "Mejor rendimiento" (overlay `ded574b5-…`) y la medición se repitió desde cero.
