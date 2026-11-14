"""CNN Base entrenada con kernels CUDA propios escritos en Python (Numba CUDA).

Este módulo implementa, sin PyTorch ni cuDNN, el paso de entrenamiento completo
de la CNN Base del Lab 3: propagación hacia adelante, retropropagación y Adam.
Cada operación es un kernel con un reparto explícito de hilos, bloques y grid.

Formatos (iguales a Keras, para poder comparar pesos)
------------------------------------------------------
* Activaciones: NHWC (lote, alto, ancho, canales), float32, contiguas.
* Pesos de convolución: (3, 3, C_in, C_out)   — formato HWIO de Keras.
* Pesos densos: (entradas, salidas).

Decisiones de diseño
--------------------
* **El canal de salida va en threadIdx.x.** En NHWC los canales son la dimensión
  contigua, así que 32 hilos consecutivos de un warp leen 32 pesos consecutivos
  (lectura *coalesced*) y escriben 32 salidas consecutivas, mientras todos leen
  el mismo píxel de entrada (difusión desde la caché).
* **Solo float32.** La MX450 ejecuta float64 a 1/32 de la velocidad de float32
  (nvidia-smi / cuda.detect: "FP32/FP64 Performance Ratio: 32"). Todas las
  constantes se escriben como float32 para que Numba no promueva a float64.
* **Kernels cortos.** En Windows la GPU está en modo WDDM: si un kernel tarda más
  de ~2 s, el sistema reinicia el controlador (TDR). Por eso el trabajo se divide
  por lotes de 64 imágenes y ningún kernel recorre el dataset completo.
* **Sin atomics donde se puede evitar.** El max-pooling 2x2 con paso 2 no tiene
  ventanas solapadas: cada píxel de entrada pertenece a una sola ventana, así que
  su retropropagación se escribe como un *gather* (un hilo por píxel de entrada).
  Los atomics solo se usan para sumar gradientes de pesos repartidos en trozos.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import cuda, float32, float64, int32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

# Tamaño de los bloques
TILE = 16            # matmul: baldosas de 16x16 en memoria compartida
HILOS_CANAL = 32     # hilos por bloque en la dimensión de canales (un warp)
HILOS_PIXEL = 8      # hilos por bloque en la dimensión de píxeles
HILOS_1D = 256       # kernels elemento a elemento

F0 = float32(0.0)
F1 = float32(1.0)


def _bloques(n: int, hilos: int) -> int:
    return (n + hilos - 1) // hilos


# =========================================================================== #
# 1. Convolución 3x3, padding 'same', paso 1, con sesgo y ReLU fusionados
# =========================================================================== #
@cuda.jit(fastmath=True)
def k_conv_forward_relu(x, w, b, salida):
    """salida[n,y,x,co] = relu(b[co] + sum_{ky,kx,ci} x[n,y+ky-1,x+kx-1,ci] * w[ky,kx,ci,co])

    Hilos:  threadIdx.x -> canal de salida co    (32 por bloque)
            threadIdx.y -> píxel (n, y, x) aplanado (8 por bloque)
    Grid:   (ceil(C_out/32), ceil(N*H*W/8))
    Un hilo calcula un valor de salida: 9*C_in multiplicaciones-suma.
    """
    co = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    p = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    N, H, W, C_in = x.shape
    C_out = w.shape[3]
    if co >= C_out or p >= N * H * W:
        return
    n = p // (H * W)
    r = p - n * H * W
    yy = r // W
    xx = r - yy * W
    acc = b[co]
    for ky in range(3):
        iy = yy + ky - 1
        if iy < 0 or iy >= H:
            continue
        for kx in range(3):
            ix = xx + kx - 1
            if ix < 0 or ix >= W:
                continue
            for ci in range(C_in):
                acc += x[n, iy, ix, ci] * w[ky, kx, ci, co]
    salida[n, yy, xx, co] = acc if acc > F0 else F0


@cuda.jit(fastmath=True)
def k_conv_backward_input(dz, w, dx):
    """Gradiente respecto a la entrada (convolución "transpuesta").

    dx[n,y,x,ci] = sum_{ky,kx,co} dz[n, y-ky+1, x-kx+1, co] * w[ky,kx,ci,co]
    Hilos: threadIdx.x -> canal de entrada ci; threadIdx.y -> píxel. Un hilo por
    elemento de dx, sin escrituras concurrentes (no requiere atomics).
    """
    ci = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    p = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    N, H, W, C_out = dz.shape
    C_in = w.shape[2]
    if ci >= C_in or p >= N * H * W:
        return
    n = p // (H * W)
    r = p - n * H * W
    yy = r // W
    xx = r - yy * W
    acc = F0
    for ky in range(3):
        oy = yy - ky + 1
        if oy < 0 or oy >= H:
            continue
        for kx in range(3):
            ox = xx - kx + 1
            if ox < 0 or ox >= W:
                continue
            for co in range(C_out):
                acc += dz[n, oy, ox, co] * w[ky, kx, ci, co]
    dx[n, yy, xx, ci] = acc


@cuda.jit(fastmath=True)
def k_conv_backward_pesos(x, dz, dw, db, posiciones_por_trozo):
    """Gradiente de pesos y sesgos, repartido en trozos de posiciones.

    dw[ky,kx,ci,co] = sum_{n,y,x} x[n,y+ky-1,x+kx-1,ci] * dz[n,y,x,co]

    Cada peso suma sobre N*H*W posiciones (hasta 262,144 en la primera capa). Un
    solo hilo por peso sería demasiado lento y dejaría la GPU casi vacía (la
    primera capa solo tiene 864 pesos), así que el trabajo se divide en tres ejes:
      threadIdx.x -> co (32)       blockIdx.x -> grupo de 32 canales de salida
      threadIdx.y -> 8 hilos que se reparten las posiciones del trozo
      blockIdx.y  -> combinación (ky, kx, ci)
      blockIdx.z  -> trozo de posiciones
    Los 8 hilos en y reducen sus sumas parciales en memoria compartida y el hilo
    y=0 suma el resultado del trozo con ``atomic.add`` (varios trozos escriben el
    mismo peso). ``dw`` y ``db`` deben llegar en cero.
    """
    parcial = cuda.shared.array(shape=(HILOS_PIXEL, HILOS_CANAL), dtype=float32)
    parcial_b = cuda.shared.array(shape=(HILOS_PIXEL, HILOS_CANAL), dtype=float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    co = cuda.blockIdx.x * HILOS_CANAL + tx
    N, H, W, C_in = x.shape
    C_out = dz.shape[3]
    kc = cuda.blockIdx.y                       # índice aplanado de (ky, kx, ci)
    ky = kc // (3 * C_in)
    kx = (kc // C_in) % 3
    ci = kc % C_in

    total = N * H * W
    inicio = cuda.blockIdx.z * posiciones_por_trozo
    fin = min(inicio + posiciones_por_trozo, total)
    acc = F0
    acc_b = F0
    if co < C_out:
        for p in range(inicio + ty, fin, HILOS_PIXEL):
            n = p // (H * W)
            r = p - n * H * W
            yy = r // W
            xx = r - yy * W
            g = dz[n, yy, xx, co]
            iy = yy + ky - 1
            ix = xx + kx - 1
            if 0 <= iy < H and 0 <= ix < W:
                acc += x[n, iy, ix, ci] * g
            acc_b += g
    parcial[ty, tx] = acc
    parcial_b[ty, tx] = acc_b
    cuda.syncthreads()
    if ty == 0 and co < C_out:
        s = F0
        sb = F0
        for k in range(HILOS_PIXEL):
            s += parcial[k, tx]
            sb += parcial_b[k, tx]
        cuda.atomic.add(dw, (ky, kx, ci, co), s)
        if kc == 0:                     # el sesgo se acumula una sola vez por trozo
            cuda.atomic.add(db, co, sb)


# =========================================================================== #
# 1b. Convolución como multiplicación de matrices implícita (implicit GEMM)
# =========================================================================== #
# Los kernels "directos" de arriba asignan un hilo por salida y leen la memoria
# global en cada multiplicación. Medidos, la retropropagación tardaba 3-4 veces
# más que la propagación: en k_conv_backward_input los hilos vecinos (ci, ci+1)
# leen pesos separados por C_out posiciones (acceso no contiguo), y
# k_conv_backward_pesos deja la GPU casi vacía en las capas con pocos pesos.
#
# Reescribiendo la convolución como producto de matrices:
#
#     salida(P x C_out) = A(P x 9*C_in) @ W(9*C_in x C_out),   P = N*H*W
#
# donde la fila p de A son los 3x3xC_in píxeles alrededor del píxel p ("im2col").
# A nunca se materializa (ocuparía 9 veces la entrada): cada bloque calcula sus
# valores al cargarlos a memoria compartida. Los pesos HWIO de Keras, vistos como
# matriz (9*C_in, C_out), ya tienen exactamente el orden que necesita W.
#
# Así, cada valor cargado en memoria compartida se reutiliza TILE=16 veces en lugar
# de leerse una vez de memoria global por cada multiplicación.

@cuda.jit(fastmath=True)
def k_conv_gemm(x, w2, sesgo, salida2, usar_sesgo, relu):
    """salida2(P, C_out) = im2col(x) @ w2 (+ sesgo) (ReLU).

    Hilos: bloques de 16x16; threadIdx.x -> canal de salida, threadIdx.y -> píxel.
    Grid:  (ceil(P/16), ceil(C_out/16)). Los píxeles van en el eje x del grid
    porque P puede superar el límite de 65,535 bloques del eje y.

    También calcula el gradiente respecto a la entrada si se le pasa dz como ``x``
    y los pesos volteados (ver ``k_voltear_pesos``) como ``w2``.
    """
    sA = cuda.shared.array(shape=(TILE, TILE), dtype=float32)
    sB = cuda.shared.array(shape=(TILE, TILE), dtype=float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    N, H, W, C_in = x.shape
    P = N * H * W
    K = 9 * C_in
    C_out = w2.shape[1]
    p = cuda.blockIdx.x * TILE + ty           # fila de la salida
    co = cuda.blockIdx.y * TILE + tx          # columna de la salida

    # Coordenadas del píxel que carga este hilo en la baldosa de A (fila p)
    n = p // (H * W)
    r = p - n * H * W
    yy = r // W
    xx = r - yy * W

    acc = F0
    for t in range((K + TILE - 1) // TILE):
        k = t * TILE + tx                     # columna de A = (ky, kx, ci)
        valor = F0
        if p < P and k < K:
            ky = k // (3 * C_in)
            kx = (k // C_in) % 3
            ci = k % C_in
            iy = yy + ky - 1
            ix = xx + kx - 1
            if 0 <= iy < H and 0 <= ix < W:
                valor = x[n, iy, ix, ci]
        sA[ty, tx] = valor
        kb = t * TILE + ty
        sB[ty, tx] = w2[kb, co] if (kb < K and co < C_out) else F0
        cuda.syncthreads()
        for j in range(TILE):
            acc += sA[ty, j] * sB[j, tx]
        cuda.syncthreads()

    if p < P and co < C_out:
        if usar_sesgo:
            acc += sesgo[co]
        if relu and acc < F0:
            acc = F0
        salida2[p, co] = acc


# Versión con "register tiling": cada bloque calcula una baldosa de 64 píxeles x
# 32 canales con 128 hilos, y cada hilo acumula 4x4 salidas en registros.
# Por cada valor leído de memoria compartida se hacen 4 multiplicaciones-suma en
# lugar de 1, lo que alivia el cuello de botella de la memoria compartida.
BM, BN, BK = 64, 32, 16      # baldosa del bloque: filas (píxeles), columnas (canales), reducción
RM, RN = 4, 4                # salidas por hilo
HILOS_RT = (BN // RN, BM // RM)   # (8, 16) = 128 hilos


@cuda.jit(fastmath=True)
def k_conv_gemm_rt(x, w2, sesgo, salida2, usar_sesgo, relu):
    """Igual que ``k_conv_gemm`` pero con register tiling 4x4.

    Hilos: (8, 16) por bloque; el hilo (tx, ty) calcula las filas ty*4..ty*4+3 y
           las columnas tx*4..tx*4+3 de la baldosa de 64x32 del bloque.
    Grid:  (ceil(P/64), ceil(C_out/32)).
    """
    sA = cuda.shared.array(shape=(BK, BM), dtype=float32)   # transpuesta: [k, fila]
    sB = cuda.shared.array(shape=(BK, BN), dtype=float32)   # [k, columna]
    acc = cuda.local.array((RM, RN), dtype=float32)
    a_reg = cuda.local.array(RM, dtype=float32)
    b_reg = cuda.local.array(RN, dtype=float32)

    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    tid = ty * (BN // RN) + tx                 # 0..127
    N, H, W, C_in = x.shape
    P = N * H * W
    K = 9 * C_in
    C_out = w2.shape[1]
    fila0 = cuda.blockIdx.x * BM
    col0 = cuda.blockIdx.y * BN

    for i in range(RM):
        for j in range(RN):
            acc[i, j] = F0

    for k0 in range(0, K, BK):
        # Carga de A: 64x16 = 1024 valores, 8 por hilo. El hilo tid carga la
        # columna k = tid % 16 de las filas (tid // 16) + 8*e.
        kk = tid % BK
        k = k0 + kk
        ky = k // (3 * C_in)
        kx = (k // C_in) % 3
        ci = k % C_in
        for e in range(BM * BK // 128):
            fila = tid // BK + e * (128 // BK)
            p = fila0 + fila
            valor = F0
            if p < P and k < K:
                n = p // (H * W)
                r = p - n * H * W
                iy = r // W + ky - 1
                ix = r % W + kx - 1
                if 0 <= iy < H and 0 <= ix < W:
                    valor = x[n, iy, ix, ci]
            sA[kk, fila] = valor
        # Carga de B: 16x32 = 512 valores, 4 por hilo (columnas contiguas)
        for e in range(BK * BN // 128):
            idx = tid + e * 128
            kb = idx // BN
            cb = idx % BN
            sB[kb, cb] = w2[k0 + kb, col0 + cb] if (k0 + kb < K and col0 + cb < C_out) else F0
        cuda.syncthreads()

        for kk2 in range(BK):
            for i in range(RM):
                a_reg[i] = sA[kk2, ty * RM + i]
            for j in range(RN):
                b_reg[j] = sB[kk2, tx * RN + j]
            for i in range(RM):
                for j in range(RN):
                    acc[i, j] += a_reg[i] * b_reg[j]
        cuda.syncthreads()

    for i in range(RM):
        p = fila0 + ty * RM + i
        if p < P:
            for j in range(RN):
                co = col0 + tx * RN + j
                if co < C_out:
                    v = acc[i, j]
                    if usar_sesgo:
                        v += sesgo[co]
                    if relu and v < F0:
                        v = F0
                    salida2[p, co] = v


@cuda.jit(fastmath=True)
def k_conv_gemm_pesos(x, dz2, dw2, filas_por_trozo):
    """dw2(9*C_in, C_out) += im2col(x)^T @ dz2(P, C_out), por trozos de P.

    La reducción recorre P = N*H*W filas (262,144 en la primera capa) mientras que
    la salida es pequeña (27 x 32 en la primera capa): con un solo bloque por
    baldosa de salida habría apenas 4 bloques para 14 multiprocesadores. Por eso
    la reducción se parte en trozos (blockIdx.z, técnica *split-K*); cada bloque
    suma su trozo en registros y lo agrega al resultado con atomic.add.

    Hilos: 16x16; threadIdx.y -> fila de dw (ky,kx,ci), threadIdx.x -> co.
    Grid:  (ceil(C_out/16), ceil(9*C_in/16), ceil(P/filas_por_trozo)).
    """
    sA = cuda.shared.array(shape=(TILE, TILE), dtype=float32)   # [fila p, k]
    sB = cuda.shared.array(shape=(TILE, TILE), dtype=float32)   # [fila p, co]
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    N, H, W, C_in = x.shape
    P = N * H * W
    K = 9 * C_in
    C_out = dz2.shape[1]
    k_salida = cuda.blockIdx.y * TILE + ty
    co = cuda.blockIdx.x * TILE + tx
    k_carga = cuda.blockIdx.y * TILE + tx     # columna de A que carga este hilo
    ky = k_carga // (3 * C_in)
    kx = (k_carga // C_in) % 3
    ci = k_carga % C_in

    inicio = cuda.blockIdx.z * filas_por_trozo
    fin = min(inicio + filas_por_trozo, P)
    acc = F0
    for base in range(inicio, fin, TILE):
        p = base + ty
        valor = F0
        g = F0
        if p < fin:
            if k_carga < K:
                n = p // (H * W)
                r = p - n * H * W
                yy = r // W
                xx = r - yy * W
                iy = yy + ky - 1
                ix = xx + kx - 1
                if 0 <= iy < H and 0 <= ix < W:
                    valor = x[n, iy, ix, ci]
            co_carga = cuda.blockIdx.x * TILE + tx
            if co_carga < C_out:
                g = dz2[p, co_carga]
        sA[ty, tx] = valor
        sB[ty, tx] = g
        cuda.syncthreads()
        for j in range(TILE):
            acc += sA[j, ty] * sB[j, tx]
        cuda.syncthreads()

    if k_salida < K and co < C_out:
        cuda.atomic.add(dw2, (k_salida, co), acc)


@cuda.jit
def k_voltear_pesos(w, w_volteado):
    """w_volteado[ky, kx, co, ci] = w[2-ky, 2-kx, ci, co].

    Con estos pesos, el gradiente respecto a la entrada es otra convolución 'same':
    dx[n,y,x,ci] = sum dz[n, y+ky-1, x+kx-1, co] * w_volteado[ky, kx, co, ci].
    Un hilo por peso.
    """
    i = cuda.grid(1)
    _, _, C_in, C_out = w.shape
    if i < w.size:
        ci = i % C_in
        co = (i // C_in) % C_out
        kx = (i // (C_in * C_out)) % 3
        ky = i // (3 * C_in * C_out)
        w_volteado[ky, kx, co, ci] = w[2 - ky, 2 - kx, ci, co]


@cuda.jit
def k_suma_columnas_trozos(X, salida, filas_por_trozo):
    """salida[j] += sum de un trozo de filas de X (gradiente del sesgo de una
    convolución: P filas x C_out columnas). Hilo por columna, trozo por blockIdx.y,
    combinados con atomic.add. ``salida`` debe llegar en cero."""
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if j >= X.shape[1]:
        return
    inicio = cuda.blockIdx.y * filas_por_trozo
    fin = min(inicio + filas_por_trozo, X.shape[0])
    s = F0
    for i in range(inicio, fin):
        s += X[i, j]
    cuda.atomic.add(salida, j, s)


# =========================================================================== #
# 2. Max-pooling 2x2 (paso 2)
# =========================================================================== #
@cuda.jit
def k_maxpool_forward(x, salida, indice):
    """Un hilo por elemento de salida. Guarda en ``indice`` (0..3) cuál de los
    cuatro píxeles de la ventana fue el máximo, para la retropropagación."""
    co = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    p = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    N, Ho, Wo, C = salida.shape
    if co >= C or p >= N * Ho * Wo:
        return
    n = p // (Ho * Wo)
    r = p - n * Ho * Wo
    oy = r // Wo
    ox = r - oy * Wo
    mejor = x[n, 2 * oy, 2 * ox, co]
    k_mejor = 0
    for k in range(1, 4):
        v = x[n, 2 * oy + k // 2, 2 * ox + k % 2, co]
        if v > mejor:
            mejor = v
            k_mejor = k
    salida[n, oy, ox, co] = mejor
    indice[n, oy, ox, co] = k_mejor


@cuda.jit
def k_maxpool_relu_backward(d_salida, indice, activacion, dz):
    """Retropropaga por el pooling y por la ReLU previa en un solo kernel.

    Un hilo por píxel de ENTRADA (gather): recibe el gradiente de su ventana solo
    si fue el máximo, y lo anula si la ReLU estaba apagada (activación = 0).
    Como las ventanas no se solapan, ningún hilo escribe donde escribe otro.
    """
    c = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    p = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    N, H, W, C = dz.shape
    if c >= C or p >= N * H * W:
        return
    n = p // (H * W)
    r = p - n * H * W
    yy = r // W
    xx = r - yy * W
    oy, ox = yy // 2, xx // 2
    g = F0
    if indice[n, oy, ox, c] == (yy % 2) * 2 + (xx % 2) and activacion[n, yy, xx, c] > F0:
        g = d_salida[n, oy, ox, c]
    dz[n, yy, xx, c] = g


# =========================================================================== #
# 3. Multiplicación de matrices con baldosas en memoria compartida
# =========================================================================== #
@cuda.jit(fastmath=True)
def k_matmul(A, B, C, trans_a, trans_b, sesgo, usar_sesgo, relu):
    """C = op(A) @ op(B) [+ sesgo] [ReLU], con op(X) = X o X^T.

    Bloques de 16x16 hilos; cada hilo calcula un elemento C[i, j]. En cada paso
    el bloque copia una baldosa de 16x16 de A y otra de B a memoria compartida y
    cada valor leído de la memoria global se reutiliza 16 veces. Grid:
    (ceil(cols/16), ceil(filas/16)). Se usa para las dos capas densas, hacia
    adelante (A @ W) y hacia atrás (A^T @ dZ para los pesos, dZ @ W^T para la entrada).
    """
    sA = cuda.shared.array(shape=(TILE, TILE), dtype=float32)
    sB = cuda.shared.array(shape=(TILE, TILE), dtype=float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    fila = cuda.blockIdx.y * TILE + ty
    col = cuda.blockIdx.x * TILE + tx
    M = A.shape[1] if trans_a else A.shape[0]
    K = A.shape[0] if trans_a else A.shape[1]
    Ncols = B.shape[0] if trans_b else B.shape[1]

    acc = F0
    for t in range((K + TILE - 1) // TILE):
        ka = t * TILE + tx
        if fila < M and ka < K:
            sA[ty, tx] = A[ka, fila] if trans_a else A[fila, ka]
        else:
            sA[ty, tx] = F0
        kb = t * TILE + ty
        if col < Ncols and kb < K:
            sB[ty, tx] = B[col, kb] if trans_b else B[kb, col]
        else:
            sB[ty, tx] = F0
        cuda.syncthreads()
        for k in range(TILE):
            acc += sA[ty, k] * sB[k, tx]
        cuda.syncthreads()

    if fila < M and col < Ncols:
        if usar_sesgo:
            acc += sesgo[col]
        if relu and acc < F0:
            acc = F0
        C[fila, col] = acc


@cuda.jit
def k_cero(x):
    """x[:] = 0 sobre un arreglo 1-D de la GPU."""
    i = cuda.grid(1)
    if i < x.size:
        x[i] = F0


@cuda.jit
def k_suma_columnas(X, salida):
    """salida[j] = sum_i X[i, j]  (gradiente del sesgo de una capa densa).
    Un hilo por columna; las capas densas tienen a lo sumo 64 filas por lote."""
    j = cuda.grid(1)
    if j < X.shape[1]:
        s = F0
        for i in range(X.shape[0]):
            s += X[i, j]
        salida[j] = s


# =========================================================================== #
# 4. Dropout, softmax + entropía cruzada y Adam
# =========================================================================== #
@cuda.jit
def k_dropout(x, mascara, tasa, estados):
    """Apaga cada valor con probabilidad ``tasa`` y escala el resto por 1/(1-tasa)
    (dropout invertido, como Keras). Un hilo por elemento, cada uno con su propio
    generador xoroshiro128+."""
    i = cuda.grid(1)
    if i < x.size:
        f, c = i // x.shape[1], i % x.shape[1]
        conservar = xoroshiro128p_uniform_float32(estados, i) >= tasa
        m = F1 / (F1 - tasa) if conservar else F0
        mascara[f, c] = m
        x[f, c] *= m


@cuda.jit
def k_relu_dropout_backward(d, activacion, mascara):
    """d *= mascara * (activacion > 0), en el lugar."""
    i = cuda.grid(1)
    if i < d.size:
        f, c = i // d.shape[1], i % d.shape[1]
        d[f, c] = d[f, c] * mascara[f, c] if activacion[f, c] > F0 else F0


@cuda.jit(fastmath=True)
def k_softmax_entropia(logits, etiquetas, d_logits, perdidas, aciertos):
    """Un hilo por imagen: softmax estable, pérdida, acierto y gradiente.
    d_logits = (softmax - one_hot) / N, el gradiente de la pérdida media."""
    n = cuda.grid(1)
    N, C = logits.shape
    if n >= N:
        return
    maximo = logits[n, 0]
    argmax = 0
    for c in range(1, C):
        if logits[n, c] > maximo:
            maximo = logits[n, c]
            argmax = c
    suma = F0
    for c in range(C):
        suma += math.exp(logits[n, c] - maximo)
    y = etiquetas[n]
    perdidas[n] = math.log(suma) - (logits[n, y] - maximo)
    aciertos[n] = 1 if argmax == y else 0
    inv_n = F1 / float32(N)
    for c in range(C):
        p = math.exp(logits[n, c] - maximo) / suma
        d_logits[n, c] = (p - (F1 if c == y else F0)) * inv_n


@cuda.jit(fastmath=True)
def k_adam(param, grad, m, v, lr_t, beta1, beta2, eps):
    """Actualización de Adam con la fórmula de Keras, un hilo por parámetro.
    ``lr_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)`` se calcula en la CPU."""
    i = cuda.grid(1)
    if i < param.size:
        g = grad[i]
        mi = beta1 * m[i] + (F1 - beta1) * g
        vi = beta2 * v[i] + (F1 - beta2) * g * g
        m[i] = mi
        v[i] = vi
        param[i] -= lr_t * mi / (math.sqrt(vi) + eps)


# =========================================================================== #
# Modelo: orquesta los kernels
# =========================================================================== #
CAPAS_CONV = [(3, 32), (32, 64), (64, 128)]
DENSAS = [(8 * 8 * 128, 128), (128, 29)]


def pesos_iniciales(seed: int = 42) -> dict[str, np.ndarray]:
    """Glorot uniforme y sesgos en cero, como Keras."""
    rng = np.random.default_rng(seed)

    def glorot(forma, fan_in, fan_out):
        limite = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limite, limite, size=forma).astype(np.float32)

    p = {}
    for i, (ci, co) in enumerate(CAPAS_CONV, 1):
        p[f'conv{i}_w'] = glorot((3, 3, ci, co), 9 * ci, 9 * co)
        p[f'conv{i}_b'] = np.zeros(co, np.float32)
    for i, (fi, fo) in enumerate(DENSAS, 1):
        p[f'dense{i}_w'] = glorot((fi, fo), fi, fo)
        p[f'dense{i}_b'] = np.zeros(fo, np.float32)
    return p


class CNNBaseNumba:
    """La CNN Base del Lab 3 sobre kernels CUDA propios.

    Todos los buffers (activaciones, gradientes, estados de Adam) se reservan una
    vez en la GPU para un tamaño de lote fijo y se reutilizan en cada paso: así el
    bucle de entrenamiento no pide memoria a la GPU ni la libera.
    """

    def __init__(self, lote: int = 64, seed: int = 42, tasa_dropout: float = 0.3,
                 implementacion: str = 'gemm_rt'):
        """``implementacion`` elige cómo se calculan las convoluciones:

        * 'directa': un hilo por salida, leyendo memoria global (versión 1).
        * 'gemm':    producto de matrices implícito con baldosas de 16x16 en
                     memoria compartida, un valor por hilo (versión 2).
        * 'gemm_rt': como 'gemm' pero cada hilo calcula 4x4 salidas en registros
                     (register tiling, versión 3, la más rápida).
        """
        assert implementacion in ('gemm_rt', 'gemm', 'directa')
        self.implementacion = implementacion
        self.lote = lote
        self.tasa = np.float32(tasa_dropout)
        self.params = {k: cuda.to_device(v) for k, v in pesos_iniciales(seed).items()}
        self.grads = {k: cuda.device_array_like(v) for k, v in self.params.items()}
        self.m = {k: cuda.to_device(np.zeros(v.shape, np.float32)) for k, v in self.params.items()}
        self.v = {k: cuda.to_device(np.zeros(v.shape, np.float32)) for k, v in self.params.items()}
        self.w_volteado = {i: cuda.device_array((3, 3, co, ci), np.float32)
                           for i, (ci, co) in enumerate(CAPAS_CONV, 1)}
        self.t = 0
        self.estados = create_xoroshiro128p_states(lote * 128, seed=seed)
        self._reservar(lote)

    def _reservar(self, N):
        f32 = np.float32
        self.N = N
        self.x = cuda.device_array((N, 64, 64, 3), f32)
        self.y = cuda.device_array(N, np.int32)
        self.a, self.pool, self.idx, self.dz = [], [], [], []
        lado = 64
        for _, co in CAPAS_CONV:
            self.a.append(cuda.device_array((N, lado, lado, co), f32))
            self.dz.append(cuda.device_array((N, lado, lado, co), f32))
            lado //= 2
            self.pool.append(cuda.device_array((N, lado, lado, co), f32))
            self.idx.append(cuda.device_array((N, lado, lado, co), np.int8))
        self.d_pool = [cuda.device_array(p.shape, f32) for p in self.pool]
        self.h = cuda.device_array((N, 128), f32)
        self.mascara = cuda.device_array((N, 128), f32)
        self.logits = cuda.device_array((N, 29), f32)
        self.d_logits = cuda.device_array((N, 29), f32)
        self.d_h = cuda.device_array((N, 128), f32)
        self.d_plano = cuda.device_array((N, 8 * 8 * 128), f32)
        self.perdidas = cuda.device_array(N, f32)
        self.aciertos = cuda.device_array(N, np.int32)
        if not hasattr(self, 'acumulador'):
            self.acumulador = cuda.to_device(np.zeros(2, np.float64))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _grid_canal_pixel(canales, pixeles):
        return (_bloques(canales, HILOS_CANAL), _bloques(pixeles, HILOS_PIXEL)), \
               (HILOS_CANAL, HILOS_PIXEL)

    def _matmul(self, A, B, C, ta=False, tb=False, sesgo=None, relu=False):
        M = A.shape[1] if ta else A.shape[0]
        Nc = B.shape[0] if tb else B.shape[1]
        grid = (_bloques(Nc, TILE), _bloques(M, TILE))
        s = sesgo if sesgo is not None else self.params['dense1_b']
        k_matmul[grid, (TILE, TILE)](A, B, C, ta, tb, s, sesgo is not None, relu)

    def _conv(self, x, w, salida, sesgo=None, relu=False):
        """Convolución 3x3 'same' de x (NHWC) con pesos w (3,3,C_in,C_out)."""
        N, H, W, _ = x.shape
        C_out = w.shape[3]
        if self.implementacion == 'directa':
            # La versión directa solo se usa hacia adelante (con sesgo y ReLU)
            g, b = self._grid_canal_pixel(C_out, N * H * W)
            k_conv_forward_relu[g, b](x, w, sesgo, salida)
            return
        P = N * H * W
        w2 = w.reshape(9 * w.shape[2], C_out)
        s = sesgo if sesgo is not None else self.params['conv1_b']
        if self.implementacion == 'gemm_rt':
            grid = (_bloques(P, BM), _bloques(C_out, BN))
            k_conv_gemm_rt[grid, HILOS_RT](x, w2, s, salida.reshape(P, C_out),
                                           sesgo is not None, relu)
        else:
            grid = (_bloques(P, TILE), _bloques(C_out, TILE))
            k_conv_gemm[grid, (TILE, TILE)](x, w2, s, salida.reshape(P, C_out),
                                            sesgo is not None, relu)

    def forward(self, entrenando: bool):
        P = self.params
        entrada = self.x
        for i, (_, co) in enumerate(CAPAS_CONV):
            a = self.a[i]
            n_pix = a.shape[0] * a.shape[1] * a.shape[2]
            self._conv(entrada, P[f'conv{i+1}_w'], a, sesgo=P[f'conv{i+1}_b'], relu=True)
            pool = self.pool[i]
            g, b = self._grid_canal_pixel(co, pool.shape[0] * pool.shape[1] * pool.shape[2])
            k_maxpool_forward[g, b](a, pool, self.idx[i])
            entrada = pool
        plano = self.pool[2].reshape(self.N, 8 * 8 * 128)   # Flatten en orden NHWC
        self._matmul(plano, P['dense1_w'], self.h, sesgo=P['dense1_b'], relu=True)
        if entrenando:
            k_dropout[_bloques(self.h.size, HILOS_1D), HILOS_1D](
                self.h, self.mascara, self.tasa, self.estados)
        self._matmul(self.h, P['dense2_w'], self.logits, sesgo=P['dense2_b'])
        k_softmax_entropia[_bloques(self.N, 64), 64](
            self.logits, self.y, self.d_logits, self.perdidas, self.aciertos)

    def backward(self):
        P, G = self.params, self.grads
        plano = self.pool[2].reshape(self.N, 8 * 8 * 128)
        # Capa densa 2: dW = h^T @ dlogits, db = suma, dh = dlogits @ W^T
        self._matmul(self.h, self.d_logits, G['dense2_w'], ta=True)
        k_suma_columnas[1, 32](self.d_logits, G['dense2_b'])
        self._matmul(self.d_logits, P['dense2_w'], self.d_h, tb=True)
        k_relu_dropout_backward[_bloques(self.d_h.size, HILOS_1D), HILOS_1D](
            self.d_h, self.h, self.mascara)
        # Capa densa 1
        self._matmul(plano, self.d_h, G['dense1_w'], ta=True)
        k_suma_columnas[1, 128](self.d_h, G['dense1_b'])
        self._matmul(self.d_h, P['dense1_w'], self.d_plano, tb=True)
        self.d_pool[2] = self.d_plano.reshape(self.pool[2].shape)

        # Bloques convolucionales, del último al primero
        for i in (2, 1, 0):
            a, dz = self.a[i], self.dz[i]
            N, H, W, C = a.shape
            g, b = self._grid_canal_pixel(C, N * H * W)
            k_maxpool_relu_backward[g, b](self.d_pool[i], self.idx[i], a, dz)

            entrada = self.x if i == 0 else self.pool[i - 1]
            C_in = entrada.shape[3]
            gw, gb = G[f'conv{i+1}_w'], G[f'conv{i+1}_b']
            k_cero[_bloques(gw.size, HILOS_1D), HILOS_1D](gw.reshape(gw.size))
            k_cero[_bloques(gb.size, HILOS_1D), HILOS_1D](gb)
            total = N * H * W
            if self.implementacion == 'directa':
                por_trozo = 2048
                grid = (_bloques(C, HILOS_CANAL), 9 * C_in, _bloques(total, por_trozo))
                k_conv_backward_pesos[grid, (HILOS_CANAL, HILOS_PIXEL)](
                    entrada, dz, gw, gb, por_trozo)
                if i > 0:
                    g, b = self._grid_canal_pixel(C_in, total)
                    k_conv_backward_input[g, b](dz, P[f'conv{i+1}_w'], self.d_pool[i - 1])
            else:
                dz2 = dz.reshape(total, C)
                por_trozo = 4096
                grid = (_bloques(C, TILE), _bloques(9 * C_in, TILE), _bloques(total, por_trozo))
                k_conv_gemm_pesos[grid, (TILE, TILE)](
                    entrada, dz2, gw.reshape(9 * C_in, C), por_trozo)
                k_suma_columnas_trozos[(_bloques(C, 32), _bloques(total, por_trozo)), 32](
                    dz2, gb, por_trozo)
                if i > 0:
                    wv = self.w_volteado[i + 1]
                    k_voltear_pesos[_bloques(wv.size, HILOS_1D), HILOS_1D](P[f'conv{i+1}_w'], wv)
                    self._conv(dz, wv, self.d_pool[i - 1])

    def adam(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-7):
        self.t += 1
        lr_t = np.float32(lr * math.sqrt(1 - beta2 ** self.t) / (1 - beta1 ** self.t))
        for k, p in self.params.items():
            p1, g1 = p.reshape(p.size), self.grads[k].reshape(p.size)
            m1, v1 = self.m[k].reshape(p.size), self.v[k].reshape(p.size)
            k_adam[_bloques(p.size, HILOS_1D), HILOS_1D](
                p1, g1, m1, v1, lr_t, np.float32(beta1), np.float32(beta2), np.float32(eps))

    # ------------------------------------------------------------------ #
    def cargar_lote(self, X: np.ndarray, y: np.ndarray, stream=0):
        """Copia un lote CPU -> GPU. Si es más corto que el reservado (último lote
        de la época), reserva de nuevo los buffers para ese tamaño.

        Si X e y están en memoria *pinned* y se pasa un ``stream``, la copia es
        asíncrona: la CPU no espera a que termine y puede seguir encolando kernels.
        """
        if len(X) != self.N:
            self._reservar(len(X))
        self.x.copy_to_device(X, stream=stream)
        self.y.copy_to_device(y, stream=stream)

    def paso_entrenamiento(self):
        self.forward(entrenando=True)
        self.backward()
        self.adam()

    def acumular_metricas(self):
        """Suma pérdida y aciertos del lote en acumuladores que viven en la GPU.
        Leerlos en cada lote obligaría a sincronizar CPU y GPU 254 veces por época."""
        k_acumular[_bloques(self.N, 64), 64](self.perdidas, self.aciertos, self.acumulador)

    def leer_y_reiniciar_metricas(self) -> tuple[float, int]:
        perdida, aciertos = self.acumulador.copy_to_host()
        k_cero_f64[1, 2](self.acumulador)
        return float(perdida), int(aciertos)

    def pesos_host(self) -> dict[str, np.ndarray]:
        return {k: v.copy_to_host() for k, v in self.params.items()}

    def evaluar(self, X: np.ndarray, y: np.ndarray, lote: int = 64):
        """Exactitud y pérdida media sin dropout."""
        n_original = self.N
        self.leer_y_reiniciar_metricas()
        for i in range(0, len(X), lote):
            self.cargar_lote(np.ascontiguousarray(X[i:i + lote]),
                             np.ascontiguousarray(y[i:i + lote], dtype=np.int32))
            self.forward(entrenando=False)
            self.acumular_metricas()
        perdida, aciertos = self.leer_y_reiniciar_metricas()
        if self.N != n_original:
            self._reservar(n_original)
        return perdida / len(X), aciertos / len(X)


@cuda.jit
def k_acumular(perdidas, aciertos, acumulador):
    """acumulador[0] += sum(perdidas), acumulador[1] += sum(aciertos). Un hilo por
    imagen; float64 para no perder precisión al sumar 16 mil valores."""
    i = cuda.grid(1)
    if i < perdidas.size:
        cuda.atomic.add(acumulador, 0, float64(perdidas[i]))
        cuda.atomic.add(acumulador, 1, float64(aciertos[i]))


@cuda.jit
def k_cero_f64(x):
    i = cuda.grid(1)
    if i < x.size:
        x[i] = 0.0


def entrenar_numba(X_train, y_train, X_val, y_val, epocas=5, lote=64, seed=42,
                   verbose=True, implementacion='gemm_rt') -> tuple[CNNBaseNumba, dict]:
    """Bucle de entrenamiento con los kernels propios.

    La CPU arma cada lote (selección aleatoria de filas) en uno de dos buffers
    *pinned* que se alternan, lo copia a la GPU de forma asíncrona y encola los
    kernels del paso sin esperar a que terminen: mientras la GPU calcula el lote i,
    la CPU ya prepara el lote i+1 (doble buffer). Antes de reutilizar un buffer se
    espera solo al evento de SU copia anterior, no a toda la GPU.

    Mide por época el tiempo total y, con eventos CUDA, el de las transferencias
    CPU->GPU y el del cómputo.
    """
    modelo = CNNBaseNumba(lote=lote, seed=seed, implementacion=implementacion)
    rng = np.random.default_rng(seed)
    stream = cuda.default_stream()
    pin_x = [cuda.pinned_array((lote, 64, 64, 3), np.float32) for _ in range(2)]
    pin_y = [cuda.pinned_array(lote, np.int32) for _ in range(2)]
    copia_lista = [None, None]
    y_train32 = y_train.astype(np.int32)

    hist = {k: [] for k in ('loss', 'accuracy', 'val_loss', 'val_accuracy',
                            't_epoca', 't_transferencia', 't_computo')}
    n = len(X_train)
    for epoca in range(1, epocas + 1):
        orden = rng.permutation(n)
        eventos = []
        cuda.synchronize()
        inicio = time.perf_counter()
        for paso, i in enumerate(range(0, n, lote)):
            idx = orden[i:i + lote]
            b = paso % 2
            if copia_lista[b] is not None:
                copia_lista[b].synchronize()       # la copia anterior de este buffer terminó
            m = len(idx)
            np.take(X_train, idx, axis=0, out=pin_x[b][:m])
            np.take(y_train32, idx, out=pin_y[b][:m])
            e0, e1, e2 = cuda.event(), cuda.event(), cuda.event()
            e0.record(stream)
            modelo.cargar_lote(pin_x[b][:m], pin_y[b][:m], stream)
            e1.record(stream)
            copia_lista[b] = e1
            modelo.paso_entrenamiento()
            modelo.acumular_metricas()
            e2.record(stream)
            eventos.append((e0, e1, e2))
        cuda.synchronize()
        perdida, aciertos = modelo.leer_y_reiniciar_metricas()
        t_transf = sum(cuda.event_elapsed_time(a, b) for a, b, _ in eventos) / 1000
        t_comp = sum(cuda.event_elapsed_time(b, c) for _, b, c in eventos) / 1000
        val_loss, val_acc = modelo.evaluar(X_val, y_val)
        cuda.synchronize()
        t = time.perf_counter() - inicio
        for k, v in zip(hist, (perdida / n, aciertos / n, val_loss, val_acc, t, t_transf, t_comp)):
            hist[k].append(v)
        if verbose:
            print(f"Época {epoca:>2}/{epocas} - {t:5.1f}s | transf {t_transf:.2f}s "
                  f"cómputo {t_comp:.2f}s - acc {aciertos / n:.4f} loss {perdida / n:.4f}"
                  f" - val_acc {val_acc:.4f} val_loss {val_loss:.4f}", flush=True)
    return modelo, hist
