"""Genera informe/cifras.tex: las tablas y cifras del informe, calculadas a partir de resultados/.

Uso:
    python -m src.cifras_latex

El informe (informe/Informe.tex) hace \\input{cifras}, así que todas las cifras salen de los CSV
y no se copian a mano. Si falta algún resultado (por ejemplo, el notebook aún no se ha ejecutado),
la cifra aparece en rojo como [pendiente] y el documento compila igual.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src import analisis as an

SALIDA = os.path.join(an.RAIZ, 'informe', 'cifras.tex')
PENDIENTE = r'\pend{pendiente}'


def _csv(nombre):
    ruta = os.path.join(an.RESULTADOS, nombre)
    return pd.read_csv(ruta) if os.path.exists(ruta) else None


def num(x, dec=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '---'
    return f'{x:,.{dec}f}'.replace(',', r'\,')


def pct(x, dec=1):
    return '---' if x is None or np.isnan(x) else f'{100 * x:.{dec}f}\\%'


def veces(x, dec=1):
    return '---' if x is None or np.isnan(x) else f'{x:.{dec}f}$\\times$'


def macro(nombre, valor):
    return f'\\newcommand{{\\{nombre}}}{{{valor}}}\n'


def tabla(columnas, filas, alineacion, separar_cada=None):
    """Tabla booktabs. separar_cada: lista de índices de fila antes de los que va \\midrule."""
    lineas = [f'\\begin{{tabular}}{{{alineacion}}}', r'\toprule', ' & '.join(columnas) + r' \\', r'\midrule']
    for i, fila in enumerate(filas):
        if separar_cada and i in separar_cada:
            lineas.append(r'\midrule')
        lineas.append(' & '.join(fila) + r' \\')
    lineas += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lineas)


ETIQUETAS = {('keras', 'cpu'): 'CPU · Keras/TF (original)', ('torch', 'cpu'): 'CPU · PyTorch',
             ('torch', 'cuda'): 'GPU · PyTorch + cuDNN', ('numba', 'cuda'): 'GPU · Numba (kernels propios)'}
ORDEN_CONFIG = [('keras', 'cpu'), ('torch', 'cpu'), ('torch', 'cuda'), ('numba', 'cuda')]


def main():
    out = ['% Archivo generado por `python -m src.cifras_latex`. No editar a mano.\n']

    # --- Lab 3 completo: esta laptop contra la máquina original ---------------------------
    aqui = an.leer_lab3_ejecutado().set_index('modelo')
    orig = an.leer_lab3_ejecutado(os.path.join(an.RAIZ, 'lab3_original', 'Lab3_Final.ipynb')).set_index('modelo')
    filas = []
    for m in aqui.index:
        filas.append([m, f"{aqui.loc[m, 'epocas']}", num(aqui.loc[m, 'minutos']),
                      num(aqui.loc[m, 'minutos'] * 60 / aqui.loc[m, 'epocas']),
                      pct(aqui.loc[m, 'test_acc'], 2),
                      f"{orig.loc[m, 'epocas']}", num(orig.loc[m, 'minutos']), pct(orig.loc[m, 'test_acc'], 2)])
    cnn = ['CNN Base', 'CNN Profunda', 'CNN Base + Aug', 'CNN Profunda + Aug']
    out.append(macro('TablaLabTres', tabla(
        ['Modelo', 'Épocas', 'Min', 's/época', 'Exactitud', 'Épocas', 'Min', 'Exactitud'], filas, 'lrrrrrrr')))
    out.append(macro('LabTresMinAqui', num(aqui.loc[cnn, 'minutos'].sum())))
    out.append(macro('LabTresMinOrig', num(orig.loc[cnn, 'minutos'].sum())))
    seg_epoca = aqui['minutos'] * 60 / aqui['epocas']
    out.append(macro('LabTresEpocaBase', num(seg_epoca['CNN Base'], 0)))
    out.append(macro('LabTresEpocaProfunda', num(seg_epoca['CNN Profunda'], 0)))

    # --- Preparación ------------------------------------------------------------------------
    rp = an.resumen_preparacion()
    sec = rp[rp.variante == 'secuencial'].iloc[0]
    filas = [['Secuencial (Lab 3)', '1', f'{sec.media_s:.2f} $\\pm$ {sec.std_s:.2f}', f'{sec.media_s:.2f}',
              '1.00$\\times$', '100\\%', num(sec.imagenes_por_s, 0)]]
    for _, r in rp[rp.variante == 'paralela'].sort_values('procesos').iterrows():
        filas.append(['Paralela', f'{r.procesos}', f'{r.media_s:.2f} $\\pm$ {r.std_s:.2f}', f'{r.con_arranque_s:.2f}',
                      veces(r.speedup, 2), pct(r.eficiencia, 0), num(r.imagenes_por_s, 0)])
    out.append(macro('TablaPreparacion', tabla(
        ['Variante', 'Procesos', 'Tiempo (s)', 'Con arranque (s)', 'Speedup', 'Eficiencia', 'Imágenes/s'],
        filas, 'lrrrrrr')))
    p8 = rp[(rp.variante == 'paralela') & (rp.procesos == 8)].iloc[0]
    out.append(macro('PrepSeq', f'{sec.media_s:.1f}'))
    out.append(macro('PrepPar', f'{p8.media_s:.1f}'))
    out.append(macro('PrepSpeedup', veces(p8.speedup, 2)))
    out.append(macro('PrepEficiencia', pct(p8.eficiencia, 0)))
    out.append(macro('PrepReps', f'{int(sec.reps)}'))
    out.append(macro('PrepIdentico', 'sí' if bool(rp.identico.all()) else r'\pend{NO}'))

    # --- Entrenamiento: corridas cortas de 3 épocas -----------------------------------------
    rt = an.resumen_entrenamiento()
    hay_keras = (rt.framework == 'keras').any()
    filas, separadores = [], []
    for m in an.ORDEN_MODELOS:
        sub = rt[rt.modelo == m]
        if sub.empty:
            continue
        separadores.append(len(filas))
        for fw, dev in ORDEN_CONFIG:
            r = sub[(sub.framework == fw) & (sub.dispositivo == dev)]
            if r.empty:
                continue
            r = r.iloc[0]
            desv = '' if np.isnan(r.epoca_std) else f' $\\pm$ {r.epoca_std:.1f}'
            filas.append([m if len(filas) == separadores[-1] else '', ETIQUETAS[(fw, dev)], f'{r.reps}',
                          f'{r.epoca_s:.1f}{desv}',
                          veces(r.speedup_epoca) if hay_keras else PENDIENTE,
                          f'{r.inferencia_s:.2f}', veces(r.speedup_inferencia) if hay_keras else PENDIENTE,
                          num(r.reloj_gpu_mhz, 0), pct(r.test_acc)])
    out.append(macro('TablaEntrenamiento', tabla(
        ['Modelo', 'Configuración', 'Corr.', 's/época', 'Speedup', 'Inferencia (s)', 'Speedup inf.', 'Reloj GPU (MHz)',
         'Exactitud'], filas, 'llrrrrrrr', separadores[1:])))

    gpu = rt[(rt.framework == 'torch') & (rt.dispositivo == 'cuda')]
    numba = rt[rt.framework == 'numba']
    torch_cpu = rt[(rt.framework == 'torch') & (rt.dispositivo == 'cpu')].set_index('modelo')
    if hay_keras:
        out.append(macro('SpeedupMin', veces(gpu.speedup_epoca.min())))
        out.append(macro('SpeedupMax', veces(gpu.speedup_epoca.max())))
        out.append(macro('SpeedupInfMin', veces(gpu.speedup_inferencia.min())))
        out.append(macro('SpeedupInfMax', veces(gpu.speedup_inferencia.max())))
        out.append(macro('SpeedupNumba', veces(numba.speedup_epoca.iloc[0], 2) if len(numba) else PENDIENTE))
    else:
        for n in ('SpeedupMin', 'SpeedupMax', 'SpeedupInfMin', 'SpeedupInfMax', 'SpeedupNumba'):
            out.append(macro(n, PENDIENTE))
    if len(numba) and 'CNN Base' in torch_cpu.index:
        out.append(macro('NumbaVsTorchCPU', veces(torch_cpu.loc['CNN Base', 'epoca_s'] / numba.epoca_s.iloc[0])))
        out.append(macro('NumbaEpoca', num(numba.epoca_s.iloc[0], 0)))
        out.append(macro('NumbaReloj', num(numba.reloj_gpu_mhz.iloc[0], 0)))
    gb = gpu.set_index('modelo')
    out.append(macro('GpuEpocaBase', num(gb.loc['CNN Base', 'epoca_s'])))
    out.append(macro('GpuEpocaProfunda', num(gb.loc['CNN Profunda', 'epoca_s'])))
    out.append(macro('GpuRelojBase', num(gb.loc['CNN Base', 'reloj_gpu_mhz'], 0)))
    out.append(macro('GpuRelojProfunda', num(gb.loc['CNN Profunda', 'reloj_gpu_mhz'], 0)))

    # Speedup conservador: contra el s/época del Lab 3 completo en esta laptop (sin límite térmico del CPU)
    kb = rt[rt.framework == 'keras'].set_index('modelo')
    out.append(macro('KerasEpocaBase', num(kb.loc['CNN Base', 'epoca_s'], 0) if 'CNN Base' in kb.index else PENDIENTE))
    out.append(macro('KerasEpocaProfunda',
                     num(kb.loc['CNN Profunda', 'epoca_s'], 0) if 'CNN Profunda' in kb.index else PENDIENTE))
    out.append(macro('SpeedupBaseLab', veces(seg_epoca['CNN Base'] / gb.loc['CNN Base', 'epoca_s'])))
    out.append(macro('SpeedupProfundaLab', veces(seg_epoca['CNN Profunda'] / gb.loc['CNN Profunda', 'epoca_s'])))

    # --- Desglose del tiempo de una época en la GPU -------------------------------------------
    dg = an.desglose_gpu(rt)
    filas = [[ETIQUETAS[(r.framework, r.dispositivo)] if 'framework' in dg else r.config, r.modelo,
              f'{r.epoca_s:.2f}', f'{r.transferencia_por_epoca_s:.2f}', f'{r.computo_por_epoca_s:.2f}',
              f'{r.resto_por_epoca_s:.2f}', pct(r.pct_transferencia)] for r in dg.itertuples()]
    out.append(macro('TablaDesglose', tabla(
        ['Configuración', 'Modelo', 'Época (s)', 'CPU$\\to$GPU (s)', 'Cómputo GPU (s)', 'Resto (s)',
         '\\% transf.'], filas, 'llrrrrr')))
    out.append(macro('TransfMaxPct', pct(dg.pct_transferencia.max())))
    out.append(macro('TransfMinPct', pct(dg.pct_transferencia.min())))

    # --- Resultados del notebook ejecutado en vivo --------------------------------------------
    comp = _csv('notebook_entrenamiento_completo.csv')
    if comp is not None:
        filas = [[r['Modelo'], f"{int(r['Épocas CPU'])}", num(r['Minutos CPU (Lab 3, esta laptop)']),
                  pct(r['Exactitud CPU'], 2), f"{int(r['Épocas GPU'])}", num(r['Minutos GPU']),
                  pct(r['Exactitud GPU'], 2), veces(r['Speedup por época'])] for _, r in comp.iterrows()]
        filas.append(['\\textbf{Total}', '', f"\\textbf{{{num(comp['Minutos CPU (Lab 3, esta laptop)'].sum())}}}", '',
                      '', f"\\textbf{{{num(comp['Minutos GPU'].sum())}}}", '', ''])
        out.append(macro('TablaCompleto', tabla(
            ['Modelo', 'Épocas', 'Min', 'Exactitud', 'Épocas', 'Min', 'Exactitud', 'Speedup/época'],
            filas, 'lrrrrrrr', [len(filas) - 1])))
        out.append(macro('CompletoMinCPU', num(comp['Minutos CPU (Lab 3, esta laptop)'].sum())))
        out.append(macro('CompletoMinGPU', num(comp['Minutos GPU'].sum())))
    else:
        out += [macro('TablaCompleto', PENDIENTE), macro('CompletoMinCPU', PENDIENTE),
                macro('CompletoMinGPU', PENDIENTE)]

    fases = _csv('speedup_por_fase.csv')
    if fases is not None:
        fases = fases.rename(columns={fases.columns[0]: 'Fase'})
        filas = [[f'\\textbf{{{r.Fase}}}' if r.Fase.startswith('Total') else r.Fase,
                  num(r['Original (s)']), num(r['Paralelo (s)']),
                  '' if pd.isna(r['Hardware paralelo']) else str(r['Hardware paralelo']),
                  veces(r['Speedup'], 2), pct(r['% del tiempo original'])] for _, r in fases.iterrows()]
        out.append(macro('TablaFases', tabla(
            ['Fase', 'Original (s)', 'Paralelo (s)', 'Hardware paralelo', 'Speedup', '\\% del original'],
            filas, 'p{5.2cm}rrp{3.3cm}rr', [len(filas) - 1])))
        f = fases.set_index('Fase')
        total = f.loc['Total del pipeline']
        entr = f.loc[[i for i in f.index if i.startswith('2.')][0]]
        prep_ = f.loc[[i for i in f.index if i.startswith('1.')][0]]
        ft, st = entr['% del tiempo original'], entr['Speedup']
        out.append(macro('SpeedupTotal', veces(total['Speedup'], 2)))
        out.append(macro('SpeedupEntrenamientoTotal', veces(st, 2)))
        out.append(macro('FraccionEntrenamiento', pct(ft)))
        out.append(macro('FraccionPrep', pct(prep_['% del tiempo original'])))
        out.append(macro('AmdahlSoloGPU', veces(1 / ((1 - ft) + ft / st), 2)))
        out.append(macro('AmdahlLimite', veces(1 / (1 - ft), 0)))
    else:
        out.append(macro('TablaFases', PENDIENTE))
        for n in ('SpeedupTotal', 'SpeedupEntrenamientoTotal', 'FraccionEntrenamiento', 'FraccionPrep',
                  'AmdahlSoloGPU', 'AmdahlLimite'):
            out.append(macro(n, PENDIENTE))

    ver = _csv('notebook_numba_versiones.csv')
    if ver is not None:
        ver = ver.rename(columns={ver.columns[0]: 'Version'})
        filas = [[r.Version.replace('×', '$\\times$'), f"{r['ms por paso (lote 64)']:.1f}",
                  f"{r['s por época (254 pasos)']:.1f}", veces(r['Mejora sobre v1'])] for _, r in ver.iterrows()]
        out.append(macro('TablaNumbaVersiones', tabla(
            ['Versión', 'ms por paso', 's por época (254 pasos)', 'Mejora sobre v1'], filas, 'lrrr')))
    else:
        out.append(macro('TablaNumbaVersiones', PENDIENTE))

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    with open(SALIDA, 'w', encoding='utf-8') as fh:
        fh.write(''.join(out))
    print(f'{len(out) - 1} macros -> {os.path.relpath(SALIDA, an.RAIZ)}')


if __name__ == '__main__':
    main()
