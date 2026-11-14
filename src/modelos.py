"""Fase 2 del pipeline: las arquitecturas del Lab 3 traducidas a PyTorch.

Las redes son las mismas del Lab 3 (Ejercicios 4 y 7). Para que la comparación
CPU/GPU mida el hardware y no un cambio de algoritmo, se replican los valores por
defecto de Keras que PyTorch no comparte:

* Inicialización: Keras usa ``glorot_uniform`` en los pesos y ceros en los sesgos;
  PyTorch usa ``kaiming_uniform`` por defecto.
* Adam: Keras usa ``epsilon=1e-7``; PyTorch, ``1e-8``.
* BatchNormalization: Keras usa ``momentum=0.99`` (peso del promedio viejo) y
  ``epsilon=1e-3``; en PyTorch el equivalente es ``momentum=0.01`` y ``eps=1e-3``.
* Formato: Keras trabaja en NHWC (lote, alto, ancho, canales) y PyTorch en NCHW.
  La transposición se hace dentro de la GPU, en ``a_nchw``.
* La salida son *logits*: ``CrossEntropyLoss`` aplica el softmax internamente, que es
  equivalente a ``Dense(softmax)`` + ``sparse_categorical_crossentropy``.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

NUM_CLASES = 29


def a_nchw(x: torch.Tensor) -> torch.Tensor:
    """(N, 64, 64, 3) -> (N, 3, 64, 64), contiguo en memoria."""
    return x.permute(0, 3, 1, 2).contiguous()


def inicializar_como_keras(modelo: nn.Module) -> nn.Module:
    for capa in modelo.modules():
        if isinstance(capa, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(capa.weight)
            if capa.bias is not None:
                nn.init.zeros_(capa.bias)
    return modelo


# --------------------------------------------------------------------------- #
# Aumento de datos (Ejercicio 7), ejecutado dentro del modelo como en Keras
# --------------------------------------------------------------------------- #
class AumentoSensato(nn.Module):
    """Equivalente de las capas de aumento de Keras del Ejercicio 7.

    RandomRotation(0.04) -> ±14.4°, RandomTranslation(0.10, 0.10),
    RandomZoom(0.12), RandomContrast(0.20), RandomBrightness(0.20) y,
    opcionalmente, RandomFlip('horizontal').

    En Keras estas capas forman parte del modelo, así que corren en el mismo
    dispositivo que el entrenamiento; aquí también. Las transformaciones geométricas
    se combinan en una sola matriz afín por imagen y se aplican con ``grid_sample``.
    ``padding_mode='border'`` replica el píxel del borde, igual que
    ``fill_mode='nearest'`` de Keras. Solo actúa en modo entrenamiento.
    """

    def __init__(self, con_espejo: bool = False, rotacion: float = 0.04,
                 traslacion: float = 0.10, zoom: float = 0.12,
                 contraste: float = 0.20, brillo: float = 0.20):
        super().__init__()
        self.con_espejo = con_espejo
        self.rotacion = rotacion * 2 * math.pi   # fracción de vuelta -> radianes
        self.traslacion = traslacion
        self.zoom = zoom
        self.contraste = contraste
        self.brillo = brillo

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x en NCHW, rango [0, 1]
        if not self.training:
            return x
        n, dev = x.shape[0], x.device

        def uniforme(a, b):
            return torch.empty(n, device=dev).uniform_(a, b)

        angulo = uniforme(-self.rotacion, self.rotacion)
        escala = uniforme(1 - self.zoom, 1 + self.zoom)
        # affine_grid usa coordenadas normalizadas en [-1, 1]: el ancho completo
        # mide 2, así que desplazar un 10% de la imagen equivale a 0.2.
        tx = uniforme(-self.traslacion, self.traslacion) * 2
        ty = uniforme(-self.traslacion, self.traslacion) * 2
        espejo = torch.ones(n, device=dev)
        if self.con_espejo:
            espejo = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)

        cos, sen = torch.cos(angulo) * escala, torch.sin(angulo) * escala
        theta = torch.stack([
            torch.stack([cos * espejo, -sen, tx], dim=1),
            torch.stack([sen * espejo, cos, ty], dim=1),
        ], dim=1)                                                    # (N, 2, 3)
        malla = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, malla, mode='bilinear', padding_mode='border',
                          align_corners=False)

        # Contraste: (x - media) * factor + media, con la media por imagen y canal
        factor = uniforme(1 - self.contraste, 1 + self.contraste).view(n, 1, 1, 1)
        media = x.mean(dim=(2, 3), keepdim=True)
        x = (x - media) * factor + media
        # Brillo: suma un desplazamiento y recorta al rango [0, 1]
        delta = uniforme(-self.brillo, self.brillo).view(n, 1, 1, 1)
        return (x + delta).clamp_(0.0, 1.0)


# --------------------------------------------------------------------------- #
# Modelo 1: CNN Base (Ejercicio 4.1)
# --------------------------------------------------------------------------- #
class CNNBase(nn.Module):
    def __init__(self, aumento: nn.Module | None = None, num_clases: int = NUM_CLASES):
        super().__init__()
        self.aumento = aumento if aumento is not None else nn.Identity()
        self.caracteristicas = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 64 -> 32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16 -> 8
        )
        self.cabeza = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_clases),
        )
        inicializar_como_keras(self)

    def forward(self, x_nhwc: torch.Tensor) -> torch.Tensor:
        x = self.aumento(a_nchw(x_nhwc))
        return self.cabeza(self.caracteristicas(x))


# --------------------------------------------------------------------------- #
# Modelo 2: CNN Profunda con BatchNorm + GlobalAveragePooling (Ejercicio 4.2)
# --------------------------------------------------------------------------- #
def bloque_conv(entrada: int, filtros: int) -> nn.Sequential:
    """Bloque doble Conv-BN-ReLU seguido de reducción y dropout."""
    capas = []
    for c_in in (entrada, filtros):
        capas += [nn.Conv2d(c_in, filtros, 3, padding=1, bias=False),
                  nn.BatchNorm2d(filtros, eps=1e-3, momentum=0.01),
                  nn.ReLU()]
    capas += [nn.MaxPool2d(2), nn.Dropout(0.25)]
    return nn.Sequential(*capas)


class CNNProfunda(nn.Module):
    def __init__(self, aumento: nn.Module | None = None, num_clases: int = NUM_CLASES):
        super().__init__()
        self.aumento = aumento if aumento is not None else nn.Identity()
        self.caracteristicas = nn.Sequential(
            bloque_conv(3, 32), bloque_conv(32, 64), bloque_conv(64, 128))
        self.cabeza = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),            # GlobalAveragePooling2D
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_clases),
        )
        inicializar_como_keras(self)

    def forward(self, x_nhwc: torch.Tensor) -> torch.Tensor:
        x = self.aumento(a_nchw(x_nhwc))
        return self.cabeza(self.caracteristicas(x))


CONSTRUCTORES = {
    'CNN Base': lambda: CNNBase(),
    'CNN Profunda': lambda: CNNProfunda(),
    'CNN Base + Aug': lambda: CNNBase(AumentoSensato()),
    'CNN Profunda + Aug': lambda: CNNProfunda(AumentoSensato()),
}


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters())
