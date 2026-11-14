# Universidad del Valle de Guatemala
**Facultad de Ingeniería**  
**Departamento de Ciencias de la Computación**  
**CC3069 - Computación Paralela y Distribuida**  
**Semestre II-2026**

## Proyecto 3: Mejorando algoritmos con CUDA

### I. Introducción
El propósito principal de este proyecto es mostrar una aplicación práctica y útil de la paralelización para optimizar y acelerar directamente un algoritmo real que ustedes ya hayan desarrollado previamente en el curso *Data Science*. Utilizando el **"Laboratorio 3. Deep Learning: Reconocimiento de Lenguaje de Señas ASL"**, el propósito fundamental es implementar sus soluciones utilizando la arquitectura del GPU con CUDA, lo que nos permitirá medir y evaluar de manera práctica la mejora real en el rendimiento de procesamiento al trasladar la carga computacional hacia el GPU.

La traducción del Lenguaje de Señas Americano (ASL) en tiempo real es una aplicación de tecnología inclusiva que busca facilitar la comunicación mediante el reconocimiento de señas realizadas frente a una cámara. Este tipo de aplicaciones requiere procesar imágenes y realizar numerosas operaciones matemáticas para identificar y clasificar las señas. Su desarrollo puede abordarse mediante programación secuencial, utilizando la CPU, o mediante programación paralela, aprovechando la capacidad de procesamiento de las GPU con tecnologías como CUDA.

---

### II. Instrucciones
El proyecto se desarrollará en equipos de **3 integrantes**, quienes deberán participar activamente en todos los pasos y documentar sus contribuciones mediante un repositorio compartido.

---

### III. Pasos

> **Nota:** Deberán documentar y presentar evidencias de todos los datos utilizados, los procedimientos realizados y las decisiones tomadas durante el proceso de análisis y diseño de la solución.

#### 1. Análisis de la solución secuencial existente
Revisen el código de la solución desarrollada previamente. Identifiquen las principales partes del programa y expliquen brevemente cuál es la función de cada una.

Descompongan el programa en sus principales operaciones, desde la recepción de las imágenes hasta la obtención del resultado. Utilicen un formato similar a la siguiente tabla para registrar los datos:

| No. | Operación identificada | Descripción de su funcionamiento |
| :--- | :--- | :--- |
| | | |

Identifiquen las operaciones que se ejecutan repetidamente y completen la siguiente tabla:

| Operación repetitiva | Datos que procesa | Cantidad de repeticiones |
| :--- | :--- | :--- |
| | | |

#### 2. Medición del tiempo de ejecución secuencial
* Ejecuten la solución secuencial desarrollada previamente y midan el tiempo utilizado para procesar un conjunto de imágenes.
* Realicen varias ejecuciones bajo las mismas condiciones para obtener resultados confiables y calcular el tiempo promedio.
* Documenten el procedimiento utilizado y presenten los resultados mediante tablas o gráficos.
* Analicen los tiempos obtenidos e identifiquen los factores que podrían influir en el rendimiento.

#### 3. Identificación de oportunidades de paralelización
Analicen las operaciones identificadas en la solución secuencial y determinen cuáles podrían ejecutarse de forma simultánea utilizando CUDA. Seleccionen y justifiquen las operaciones que proponen paralelizar, explicando cómo podrían beneficiarse del procesamiento en la GPU.

#### 4. Diseño del algoritmo paralelo
Diseñen una propuesta de algoritmo paralelo utilizando CUDA, tomando como base las operaciones seleccionadas en el paso anterior. Determinen cómo se distribuirá el trabajo entre los hilos, bloques y grids de la GPU.

#### 5. Implementación con CUDA Python
* Implementen el algoritmo paralelo diseñado en la etapa anterior utilizando CUDA Python.
* Ejecuten la solución paralela desarrollada con CUDA.
* Midan y registren los tiempos de ejecución, utilizando los mismos datos y condiciones empleados en la evaluación secuencial.
* Realicen varias pruebas y calculen el tiempo promedio, considerando tanto el procesamiento en la GPU como las transferencias de datos entre CPU y GPU.

#### 6. Análisis y conclusiones
* Comparen los resultados de ambas implementaciones mediante tablas y gráficos.
* Calculen la aceleración (*speedup*) obtenida y analicen si la paralelización permitió reducir los tiempos de ejecución.
* Expliquen las posibles causas de las diferencias de rendimiento.

---

### IV. Entregables

#### 1. Código (Scripts o Jupyter Notebook)
* Un archivo `.py` o `.ipynb` documentado que contenga el código original secuencial y la nueva versión optimizada.
* El código debe evidenciar claramente la separación del flujo de trabajo: la fase de preparación de datos y la configuración para el uso de CUDA en la fase de entrenamiento del modelo.

#### 2. Informe de Rendimiento y Análisis (Archivo PDF)
* Tabla Comparativa de Tiempos.
* Cálculo de *Speedup* para cada fase.
* Capturas de pantalla del uso de recursos del sistema (Administrador de Tareas, `nvidia-smi` o similar) durante las ejecuciones, demostrando la carga en los distintos núcleos del CPU y la saturación de la memoria/núcleos del GPU.
* **Discusión Técnica:** Un análisis respondiendo a las siguientes preguntas:
  * ¿Por qué es más eficiente procesar las imágenes/archivos con los múltiples núcleos del CPU y delegar exclusivamente la matemática de la red neuronal al GPU?
  * ¿Cómo CUDA permitió reducir o cambiar el tiempo de procesamiento?
  * ¿Qué factores justifican o limitan la utilización de CUDA?

---

### V. Evaluación

* **(40 puntos) Refactorización de Código:** Los scripts de Python actualizados que evidencien la correcta separación del pipeline explicado estrictamente para la preparación de imágenes y CUDA estrictamente para el entrenamiento del modelo.
* **(30 puntos) Evidencia de Ejecución:** Logs y capturas de pantalla de los monitores de recursos que demuestren que el entorno reconoció y utilizó correctamente el hardware asignado en cada fase.
* **(30 puntos) Reporte de Análisis:** Un reporte corto donde calculen la métrica de *Speedup* alcanzada en su prueba. Discutan técnicamente por qué la manipulación de archivos se mantuvo en el CPU mientras que la matemática de la red neuronal se delegó al GPU y qué funciones / librerías adicionales hicieron uso que permiten trabajar con el GPU.

> **Nota adicional:**  
> Tienen la opción de proponer la paralelización y aceleración de otro laboratorio o proyecto previamente desarrollado en su curso. Para habilitar esta opción, deberán presentar su propuesta a más tardar el siguiente día de clases, definiendo claramente los pasos equivalentes a realizar para su caso específico (incluida su entrega que realiza el trabajo en memoria individual, la separación de la carga de datos y la ejecución del componente pesado en GPU con CUDA).