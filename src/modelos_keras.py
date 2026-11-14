"""Línea base: los modelos Keras/TensorFlow del Lab 3, sin cambios de arquitectura.

Es la solución original. TensorFlow >= 2.11 no usa la GPU en Windows nativo, así
que estos modelos siempre entrenan en la CPU (con oneDNN y todos los hilos).

La única diferencia con el notebook original es que el número de épocas se puede
fijar sin EarlyStopping, para comparar contra la GPU con la misma cantidad de trabajo.
"""
from __future__ import annotations

import time

import numpy as np


def _keras():
    from tensorflow import keras            # import diferido: TF tarda en cargar
    from tensorflow.keras import layers
    return keras, layers


def construir_cnn_base(forma=(64, 64, 3), num_clases=29, aumento=None):
    keras, layers = _keras()
    capas = [layers.Input(shape=forma)]
    if aumento is not None:
        capas.append(aumento)
    capas += [
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_clases, activation='softmax'),
    ]
    return keras.Sequential(capas, name='CNN_Base')


def _bloque_conv(x, filtros):
    _, layers = _keras()
    for _ in range(2):
        x = layers.Conv2D(filtros, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(0.25)(x)


def construir_cnn_profunda(forma=(64, 64, 3), num_clases=29, aumento=None):
    keras, layers = _keras()
    entrada = layers.Input(shape=forma)
    x = aumento(entrada) if aumento is not None else entrada
    for filtros in (32, 64, 128):
        x = _bloque_conv(x, filtros)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    salida = layers.Dense(num_clases, activation='softmax')(x)
    return keras.Model(entrada, salida, name='CNN_Profunda')


def capas_aumento(con_espejo=False):
    keras, layers = _keras()
    capas = [layers.RandomFlip('horizontal')] if con_espejo else []
    capas += [
        layers.RandomRotation(0.04, fill_mode='nearest'),
        layers.RandomTranslation(0.10, 0.10, fill_mode='nearest'),
        layers.RandomZoom(0.12, fill_mode='nearest'),
        layers.RandomContrast(0.20),
        layers.RandomBrightness(0.20, value_range=(0.0, 1.0)),
    ]
    return keras.Sequential(capas, name='aumento')


CONSTRUCTORES = {
    'CNN Base': lambda: construir_cnn_base(),
    'CNN Profunda': lambda: construir_cnn_profunda(),
    'CNN Base + Aug': lambda: construir_cnn_base(aumento=capas_aumento()),
    'CNN Profunda + Aug': lambda: construir_cnn_profunda(aumento=capas_aumento()),
}


def entrenar_keras(nombre, X_train, y_train, X_val, y_val, epocas=20, lote=64,
                   paciencia=None, seed=42, verbose=2):
    """Compila y entrena como en el Lab 3. Devuelve (modelo, history, tiempos por época)."""
    keras, _ = _keras()
    keras.utils.set_random_seed(seed)
    modelo = CONSTRUCTORES[nombre]()
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    class Cronometro(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self._t = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            tiempos.append(time.perf_counter() - self._t)

    tiempos = []
    callbacks = [Cronometro()]
    if paciencia is not None:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=paciencia, restore_best_weights=True))
    historial = modelo.fit(X_train, y_train, validation_data=(X_val, y_val),
                           epochs=epocas, batch_size=lote, callbacks=callbacks,
                           verbose=verbose)
    return modelo, historial, np.asarray(tiempos)
