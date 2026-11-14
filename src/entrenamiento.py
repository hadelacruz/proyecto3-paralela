"""Bucle de entrenamiento instrumentado para CPU y GPU (PyTorch).

Reproduce ``modelo.fit(...)`` del Lab 3: Adam, entropía cruzada, lotes de 64,
barajado en cada época, validación al final de cada época y, opcionalmente,
EarlyStopping sobre la exactitud de validación restaurando los mejores pesos.

Medición de tiempos
-------------------
Las llamadas a CUDA son asíncronas: la CPU encola el trabajo y sigue sin esperar.
Medir con ``time.perf_counter()`` sin sincronizar mediría solo el encolado. Por eso:

* el tiempo de cada época se toma después de ``torch.cuda.synchronize()``, y
* el desglose usa ``torch.cuda.Event``, que registra marcas de tiempo *en la GPU*:
  una antes y otra después de copiar cada lote (transferencia CPU->GPU) y otra al
  terminar el paso de entrenamiento (cómputo). Al final de la época se suman.

Los datos se quedan en la RAM del CPU y viajan a la GPU lote por lote, desde
memoria *pinned* (bloqueada en RAM), que el controlador puede copiar por DMA sin
una copia intermedia. El conjunto de entrenamiento ocupa ~800 MB en float32 y la
MX450 tiene 2 GB, así que la transferencia por lotes es además la opción segura.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


@dataclass
class Historial:
    dispositivo: str
    loss: list = field(default_factory=list)
    accuracy: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_accuracy: list = field(default_factory=list)
    t_epoca: list = field(default_factory=list)          # s, reloj de pared
    t_transferencia: list = field(default_factory=list)  # s, eventos CUDA (solo GPU)
    t_computo: list = field(default_factory=list)        # s, eventos CUDA (solo GPU)
    mejor_epoca: int | None = None

    def como_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def fijar_semilla(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def a_tensores(X: np.ndarray, y: np.ndarray, pinned: bool) -> tuple[torch.Tensor, torch.Tensor]:
    Xt = torch.from_numpy(np.ascontiguousarray(X))
    yt = torch.from_numpy(y.astype(np.int64))
    if pinned:
        Xt, yt = Xt.pin_memory(), yt.pin_memory()
    return Xt, yt


@torch.no_grad()
def evaluar(modelo: nn.Module, X: torch.Tensor, y: torch.Tensor, dispositivo: torch.device,
            lote: int = 256) -> tuple[float, float, np.ndarray]:
    """Devuelve (pérdida media, exactitud, predicciones) sin aumento ni dropout."""
    modelo.eval()
    perdida_total, aciertos, preds = 0.0, 0, []
    criterio = nn.CrossEntropyLoss(reduction='sum')
    for i in range(0, len(X), lote):
        xb = X[i:i + lote].to(dispositivo, non_blocking=True)
        yb = y[i:i + lote].to(dispositivo, non_blocking=True)
        logits = modelo(xb)
        perdida_total += criterio(logits, yb).item()
        pred = logits.argmax(dim=1)
        aciertos += (pred == yb).sum().item()
        preds.append(pred.cpu())
    return perdida_total / len(X), aciertos / len(X), torch.cat(preds).numpy()


def entrenar(modelo: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray, dispositivo: str = 'cuda',
             epocas: int = 20, lote: int = 64, paciencia: int | None = None,
             seed: int = 42, verbose: bool = True) -> Historial:
    """Entrena ``modelo`` y devuelve el historial con métricas y tiempos por época.

    ``paciencia=None`` desactiva el EarlyStopping: es lo que se usa en los
    benchmarks, para que CPU y GPU ejecuten exactamente el mismo número de épocas.
    """
    dev = torch.device(dispositivo)
    es_gpu = dev.type == 'cuda'
    fijar_semilla(seed)

    modelo = modelo.to(dev)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3, eps=1e-7)
    criterio = nn.CrossEntropyLoss()

    Xtr, ytr = a_tensores(X_train, y_train, pinned=es_gpu)
    Xva, yva = a_tensores(X_val, y_val, pinned=es_gpu)
    n = len(Xtr)
    generador = torch.Generator().manual_seed(seed)

    hist = Historial(dispositivo=str(dev))
    mejor_val, mejores_pesos, sin_mejora = -1.0, None, 0

    for epoca in range(1, epocas + 1):
        modelo.train()
        orden = torch.randperm(n, generator=generador)
        perdida_acum = torch.zeros((), device=dev)
        aciertos_acum = torch.zeros((), device=dev, dtype=torch.long)
        eventos = []

        if es_gpu:
            torch.cuda.synchronize()
        inicio = time.perf_counter()

        for i in range(0, n, lote):
            idx = orden[i:i + lote]
            # El índice aleatorio produce una copia nueva en RAM normal; se vuelve
            # a fijar (pin) para que la copia a la GPU sea por DMA.
            xb_cpu, yb_cpu = Xtr[idx], ytr[idx]
            if es_gpu:
                xb_cpu, yb_cpu = xb_cpu.pin_memory(), yb_cpu.pin_memory()
                e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
                e0.record()
            xb = xb_cpu.to(dev, non_blocking=True)
            yb = yb_cpu.to(dev, non_blocking=True)
            if es_gpu:
                e1.record()

            optimizador.zero_grad(set_to_none=True)
            logits = modelo(xb)
            perdida = criterio(logits, yb)
            perdida.backward()
            optimizador.step()

            # Se acumula en el dispositivo: llamar a .item() en cada lote obligaría
            # a sincronizar CPU y GPU 254 veces por época.
            perdida_acum += perdida.detach() * len(yb)
            aciertos_acum += (logits.argmax(dim=1) == yb).sum()
            if es_gpu:
                e2.record()
                eventos.append((e0, e1, e2))

        if es_gpu:
            torch.cuda.synchronize()
        t_entrenamiento = time.perf_counter() - inicio

        val_loss, val_acc, _ = evaluar(modelo, Xva, yva, dev)
        if es_gpu:
            torch.cuda.synchronize()
        t_epoca = time.perf_counter() - inicio

        hist.loss.append(perdida_acum.item() / n)
        hist.accuracy.append(aciertos_acum.item() / n)
        hist.val_loss.append(val_loss)
        hist.val_accuracy.append(val_acc)
        hist.t_epoca.append(t_epoca)
        if es_gpu:
            hist.t_transferencia.append(sum(a.elapsed_time(b) for a, b, _ in eventos) / 1000)
            hist.t_computo.append(sum(b.elapsed_time(c) for _, b, c in eventos) / 1000)

        if verbose:
            extra = (f" | transf {hist.t_transferencia[-1]:.2f}s cómputo {hist.t_computo[-1]:.2f}s"
                     if es_gpu else '')
            print(f"Época {epoca:>2}/{epocas} - {t_epoca:5.1f}s (train {t_entrenamiento:5.1f}s){extra}"
                  f" - acc {hist.accuracy[-1]:.4f} loss {hist.loss[-1]:.4f}"
                  f" - val_acc {val_acc:.4f} val_loss {val_loss:.4f}", flush=True)

        if paciencia is not None:
            if val_acc > mejor_val:
                mejor_val, sin_mejora = val_acc, 0
                hist.mejor_epoca = epoca
                mejores_pesos = copy.deepcopy(modelo.state_dict())
            else:
                sin_mejora += 1
                if sin_mejora >= paciencia:
                    if verbose:
                        print(f"Early stopping en la época {epoca}; se restauran los pesos "
                              f"de la época {hist.mejor_epoca}.")
                    break

    if mejores_pesos is not None:
        modelo.load_state_dict(mejores_pesos)
    return hist
