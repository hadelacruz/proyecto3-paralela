"""Paso 6: tablas, speedups y gráficas a partir de los CSV de ``resultados/``.

Convenciones de las gráficas
----------------------------
* Cada configuración tiene un color fijo en todas las figuras (paleta categórica
  validada para daltonismo, en orden fijo): el color identifica la configuración,
  nunca su posición en un ranking.
* Una sola escala vertical por gráfica: cuando dos magnitudes tienen unidades
  distintas (p. ej. uso de GPU en % y frecuencia en MHz) se dibujan en paneles
  separados que comparten el eje del tiempo.
* El texto va en tinta neutra; los valores se rotulan solo donde ayudan a leer.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTADOS = os.environ.get('P3_RESULTADOS', os.path.join(RAIZ, 'resultados'))
FIGURAS = os.path.join(RAIZ, 'informe', 'figuras')

# --------------------------------------------------------------------------- #
# Estilo
# --------------------------------------------------------------------------- #
SUPERFICIE, TINTA, TINTA_SUAVE, TINTA_TENUE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
REJILLA, BORDE = '#e1e0d9', '#c3c2b7'

CONFIGS = {   # (framework, dispositivo) -> etiqueta y color (orden fijo de la paleta)
    ('keras', 'cpu'): ('CPU · Keras/TF (original)', '#2a78d6'),
    ('torch', 'cuda'): ('GPU · PyTorch + cuDNN', '#eb6834'),
    ('numba', 'cuda'): ('GPU · Numba (kernels propios)', '#1baf7a'),
    ('torch', 'cpu'): ('CPU · PyTorch', '#eda100'),
}
ORDEN_MODELOS = ['CNN Base', 'CNN Profunda', 'CNN Base + Aug', 'CNN Profunda + Aug']
RAMPA_AZUL = ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#0d366b']


def estilo():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': SUPERFICIE, 'axes.facecolor': SUPERFICIE,
        'savefig.facecolor': SUPERFICIE, 'axes.edgecolor': BORDE,
        'axes.labelcolor': TINTA_SUAVE, 'text.color': TINTA,
        'xtick.color': TINTA_TENUE, 'ytick.color': TINTA_TENUE,
        'grid.color': REJILLA, 'axes.grid': True, 'grid.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10,
        'axes.titlesize': 12, 'axes.titlelocation': 'left', 'axes.titlepad': 10,
        'legend.frameon': False, 'figure.dpi': 110, 'savefig.dpi': 160,
        'savefig.bbox': 'tight',
    })


def guardar(fig, nombre):
    os.makedirs(FIGURAS, exist_ok=True)
    fig.savefig(os.path.join(FIGURAS, f'{nombre}.png'))


def _etiqueta(fw, dev):
    return CONFIGS[(fw, dev)][0]


def _color(fw, dev):
    return CONFIGS[(fw, dev)][1]


# --------------------------------------------------------------------------- #
# Fase 1: preparación
# --------------------------------------------------------------------------- #
def resumen_preparacion() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RESULTADOS, 'preparacion.csv'))
    g = (df.groupby(['variante', 'procesos'])
           .agg(media_s=('segundos', 'mean'), std_s=('segundos', 'std'),
                con_arranque_s=('segundos_con_arranque', 'mean'),
                reps=('rep', 'count'), identico=('identico', 'all'))
           .reset_index())
    base = g.loc[g.variante == 'secuencial', 'media_s'].iloc[0]
    g['speedup'] = base / g['media_s']
    g['speedup_con_arranque'] = base / g['con_arranque_s']
    g['eficiencia'] = g['speedup'] / g['procesos']
    g['imagenes_por_s'] = 23_200 / g['media_s']
    return g


def grafica_escalamiento_preparacion(res: pd.DataFrame):
    """Speedup de la preparación contra número de procesos, con la recta ideal."""
    import matplotlib.pyplot as plt
    par = res[res.variante == 'paralela'].sort_values('procesos')
    fig, eje = plt.subplots(figsize=(7.5, 4.2))
    p = par['procesos'].to_numpy()
    eje.plot(p, p, color=TINTA_TENUE, linewidth=1.2, linestyle='--')
    eje.annotate('ideal (lineal)', xy=(p[-1], p[-1]), xytext=(-4, 4),
                 textcoords='offset points', ha='right', color=TINTA_SUAVE, fontsize=9)
    eje.plot(p, par['speedup'], color=CONFIGS[('keras', 'cpu')][1], linewidth=2,
             marker='o', markersize=8, markeredgecolor=SUPERFICIE, markeredgewidth=2)
    for x, s in zip(p, par['speedup']):
        eje.annotate(f'{s:.2f}×', xy=(x, s), xytext=(0, -16), textcoords='offset points',
                     ha='center', color=TINTA, fontsize=9)
    eje.axvline(4, color=REJILLA, linewidth=1)
    eje.annotate('4 núcleos físicos', xy=(4, 0.4), xytext=(4, 4), textcoords='offset points',
                 color=TINTA_SUAVE, fontsize=9)
    eje.set_xticks(p)
    eje.set_xlabel('Procesos')
    eje.set_ylabel('Speedup sobre la versión secuencial')
    eje.set_ylim(0, max(p) + 0.5)
    eje.set_title('Preparación de 23,200 imágenes: speedup con varios procesos')
    plt.tight_layout()
    return fig


def mapa_nucleos(csvs: dict[str, str]):
    """Mapa de calor del uso de cada hilo lógico en el tiempo, un panel por fase."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    rampa = LinearSegmentedColormap.from_list('azul', RAMPA_AZUL)
    fig, ejes = plt.subplots(len(csvs), 1, figsize=(9, 1.9 * len(csvs) + 0.6),
                             squeeze=False)
    for eje, (titulo, ruta) in zip(ejes[:, 0], csvs.items()):
        d = pd.read_csv(ruta)
        nucleos = [c for c in d.columns if re.fullmatch(r'cpu\d+', c)]
        img = eje.imshow(d[nucleos].T.to_numpy(), aspect='auto', cmap=rampa, vmin=0, vmax=100,
                         extent=[d['t'].iloc[0], d['t'].iloc[-1], len(nucleos) - 0.5, -0.5],
                         interpolation='nearest')
        eje.set_yticks(range(len(nucleos)), [f'hilo {i}' for i in range(len(nucleos))], fontsize=8)
        eje.set_title(f"{titulo} — uso medio {d['cpu_total'].mean():.0f}%", fontsize=11)
        eje.grid(False)
    ejes[-1, 0].set_xlabel('Tiempo (s)')
    fig.colorbar(img, ax=ejes[:, 0], shrink=0.8, label='% de uso')
    return fig


# --------------------------------------------------------------------------- #
# Fase 2 y 3: entrenamiento e inferencia
# --------------------------------------------------------------------------- #
def resumen_entrenamiento() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RESULTADOS, 'entrenamiento.csv'))
    g = (df.groupby(['framework', 'dispositivo', 'modelo'])
           .agg(reps=('rep', 'count'), epocas=('epocas', 'first'),
                epoca_s=('t_epoca_media_sin_1ra_s', 'mean'),
                epoca_std=('t_epoca_media_sin_1ra_s', 'std'),
                primera_epoca_s=('t_primera_epoca_s', 'mean'),
                transferencia_s=('t_transferencia_s', 'mean'),
                computo_gpu_s=('t_computo_gpu_s', 'mean'),
                inferencia_s=('t_inferencia_test_s', 'mean'),
                inferencia_std=('t_inferencia_test_s', 'std'),
                test_acc=('test_accuracy', 'mean'),
                cpu_pct=('cpu_media_pct', 'mean'), gpu_pct=('gpu_util_media_pct', 'mean'),
                gpu_mem_mb=('gpu_mem_max_mb', 'max'))
           .reset_index())
    g['config'] = [_etiqueta(f, d) for f, d in zip(g.framework, g.dispositivo)]
    # Las transferencias se suman sobre todas las épocas: se expresan por época
    g['transferencia_por_epoca_s'] = g['transferencia_s'] / g['epocas']
    g['computo_por_epoca_s'] = g['computo_gpu_s'] / g['epocas']
    base = g[(g.framework == 'keras')].set_index('modelo')
    g['speedup_epoca'] = [base.loc[m, 'epoca_s'] / e if m in base.index else np.nan
                          for m, e in zip(g.modelo, g.epoca_s)]
    g['speedup_inferencia'] = [base.loc[m, 'inferencia_s'] / e if m in base.index else np.nan
                               for m, e in zip(g.modelo, g.inferencia_s)]
    g['reloj_gpu_mhz'] = [_reloj_mediano(f, d, m) for f, d, m in zip(g.framework, g.dispositivo, g.modelo)]
    g['orden'] = g.modelo.map({m: i for i, m in enumerate(ORDEN_MODELOS)})
    return g.sort_values(['orden', 'framework']).drop(columns='orden').reset_index(drop=True)


def _reloj_mediano(framework, dispositivo, modelo) -> float:
    """Mediana del reloj SM de la GPU mientras estuvo ocupada (>50 %), en todas las corridas.

    La MX450 baja su reloj por el límite de potencia de la laptop, así que el tiempo por época
    depende de cuánto duró la corrida: este dato acompaña a cada medición de GPU.
    """
    if dispositivo != 'cuda':
        return np.nan
    nombre = modelo.replace(' ', '').replace('+', '_')
    patron = os.path.join(RESULTADOS, 'monitor', f'train_{framework}_{dispositivo}_{nombre}_rep*.csv')
    relojes = []
    for ruta in glob.glob(patron):
        d = pd.read_csv(ruta)
        util = pd.to_numeric(d['gpu_utilization_gpu'], errors='coerce')
        relojes.append(pd.to_numeric(d['gpu_clocks_sm'], errors='coerce')[util > 50])
    return float(pd.concat(relojes).median()) if relojes else np.nan


def grafica_tiempo_epoca(res: pd.DataFrame, columna='epoca_s', titulo=None, eje_x=None):
    """Barras horizontales agrupadas por modelo, una barra por configuración."""
    import matplotlib.pyplot as plt
    configs = [c for c in CONFIGS if ((res.framework == c[0]) & (res.dispositivo == c[1])).any()]
    modelos = [m for m in ORDEN_MODELOS if m in set(res.modelo)]
    alto = 0.8 / len(configs)
    fig, eje = plt.subplots(figsize=(9, 1.0 + 0.32 * len(configs) * len(modelos) + 0.4 * len(modelos)))
    maximo = res[columna].max()
    for j, (fw, dev) in enumerate(configs):
        sub = res[(res.framework == fw) & (res.dispositivo == dev)].set_index('modelo')
        for i, m in enumerate(modelos):
            if m not in sub.index:
                continue
            y = i + j * alto - 0.4 + alto / 2
            v = sub.loc[m, columna]
            eje.barh(y, v, height=alto * 0.86, color=_color(fw, dev),
                     label=_etiqueta(fw, dev) if i == 0 or m == modelos[0] else None)
            texto = f'{v:.2f}×' if columna == '_v' else f'{v:.1f} s'
            eje.text(v + maximo * 0.01, y, texto, va='center', fontsize=8.5, color=TINTA)
    eje.set_yticks(range(len(modelos)), modelos)
    eje.invert_yaxis()
    eje.set_xlim(0, maximo * 1.15)
    eje.set_xlabel(eje_x or 'Segundos por época (media de las épocas 2 en adelante)')
    eje.grid(axis='y', visible=False)
    manijas, nombres = eje.get_legend_handles_labels()
    unicos = dict(zip(nombres, manijas))
    eje.legend(unicos.values(), unicos.keys(), loc='lower right', fontsize=9)
    eje.set_title(titulo or 'Tiempo por época de entrenamiento (menos es mejor)')
    plt.tight_layout()
    return fig


def grafica_speedup(res: pd.DataFrame, columna='speedup_epoca', titulo=None):
    import matplotlib.pyplot as plt
    sub = res[(res.framework != 'keras') & res[columna].notna()]
    return grafica_tiempo_epoca(
        sub.assign(_v=sub[columna]), columna='_v',
        titulo=titulo or 'Speedup sobre la línea base Keras/TF en CPU (más es mejor)',
        eje_x='Speedup (×). 1× = igual de rápido que la CPU')


def desglose_gpu(res: pd.DataFrame) -> pd.DataFrame:
    """Por época: transferencias CPU->GPU, cómputo en GPU y el resto (Python,
    armado de lotes, validación)."""
    g = res[res.dispositivo == 'cuda'].copy()
    g['resto_por_epoca_s'] = (g['epoca_s'] - g['transferencia_por_epoca_s']
                              - g['computo_por_epoca_s']).clip(lower=0)
    g['pct_transferencia'] = g['transferencia_por_epoca_s'] / g['epoca_s']
    return g[['config', 'modelo', 'epoca_s', 'transferencia_por_epoca_s',
              'computo_por_epoca_s', 'resto_por_epoca_s', 'pct_transferencia']]


def linea_tiempo_gpu(ruta_csv: str, titulo: str):
    """Uso de GPU y frecuencia del reloj en paneles separados (unidades distintas)."""
    import matplotlib.pyplot as plt
    d = pd.read_csv(ruta_csv)
    num = lambda c: pd.to_numeric(d[c], errors='coerce')
    fig, ejes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    series = [('gpu_utilization_gpu', 'Uso GPU (%)', (0, 105)),
              ('gpu_clocks_sm', 'Reloj SM (MHz)', (0, None)),
              ('gpu_temperature_gpu', 'Temperatura (°C)', (None, None))]
    for eje, (col, etiqueta, lim) in zip(ejes, series):
        eje.plot(d['t'], num(col), color=CONFIGS[('torch', 'cuda')][1], linewidth=2)
        eje.set_ylabel(etiqueta)
        eje.set_ylim(*lim)
    ejes[0].plot(d['t'], d['cpu_total'], color=CONFIGS[('keras', 'cpu')][1], linewidth=2)
    ejes[0].legend(['GPU', 'CPU (media de los 8 hilos)'], loc='lower right', fontsize=8.5)
    ejes[-1].set_xlabel('Tiempo (s)')
    ejes[0].set_title(titulo)
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Lab 3 completo (línea base de extremo a extremo en CPU)
# --------------------------------------------------------------------------- #
def leer_lab3_ejecutado(ruta=None) -> pd.DataFrame:
    """Tiempos y exactitudes del notebook del Lab 3 ejecutado en esta máquina."""
    ruta = ruta or os.path.join(RESULTADOS, 'lab3_reproducible_ejecutado.ipynb')
    nb = json.load(open(ruta, encoding='utf-8'))
    texto = ''
    for celda in nb['cells']:
        for salida in celda.get('outputs', []):
            t = salida.get('text') or salida.get('data', {}).get('text/plain', '')
            texto += ''.join(t) + '\n'
    filas = [{'modelo': m, 'minutos': float(t), 'epocas': int(e)}
             for m, t, e in re.findall(r'--- (.+?) entrenado en ([\d.]+) min \((\d+) épocas\) ---', texto)]
    tabla = pd.DataFrame(filas)
    exact = dict(re.findall(r'^(CNN [\w +]+?|Red Densa \(MLP\))\s{2,}[\d,]+\s+\d+\s+[\d.]+\s+[\d.]+%\s+([\d.]+)%',
                            texto, flags=re.M))
    aug = dict(re.findall(r'^(CNN [\w +]+?)\s{2,}([\d.]+)%\s+[\d.]+%\s+[-+\d.]+%\s+\d+\s+[\d.]+$',
                          texto, flags=re.M))
    exact.update(aug)
    tabla['test_acc'] = tabla['modelo'].map(lambda m: float(exact[m]) / 100 if m in exact else np.nan)
    return tabla


def leer_historial(framework, dispositivo, modelo, rep=1) -> dict:
    etiqueta = f"{framework}_{dispositivo}_{modelo.replace(' ', '').replace('+', '_')}"
    with open(os.path.join(RESULTADOS, 'monitor', f'hist_{etiqueta}_rep{rep}.json'),
              encoding='utf-8') as f:
        return json.load(f)


def ruta_monitor(patron: str) -> str | None:
    rutas = sorted(glob.glob(os.path.join(RESULTADOS, 'monitor', patron)))
    return rutas[0] if rutas else None
