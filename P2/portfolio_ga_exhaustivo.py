"""
portfolio_ga_exhaustivo.py
==========================
Pruebas exhaustivas del AG para selección de cartera (Hillier & Lieberman Caso 12.1).
Barre combinaciones de parámetros y genera bitácoras detalladas en CSV y TXT.
"""

import numpy as np
import csv
import os
import time
import itertools
from datetime import datetime

# ============================================================
#  DATOS DEL PROBLEMA
# ============================================================
rng = np.random.default_rng()   # sin semilla fija → mayor variabilidad

ACCIONES  = ['BB', 'LOP', 'ILI', 'HEAL', 'QUI', 'AUA']
N_ACCIONES = len(ACCIONES)

RENDIMIENTOS = np.array([0.20, 0.42, 1.00, 0.50, 0.46, 0.30])

COV = np.array([
    [ 0.032,  0.005,  0.030, -0.031, -0.027,  0.010],
    [ 0.005,  0.100,  0.085, -0.070, -0.050,  0.020],
    [ 0.030,  0.085,  0.333, -0.110, -0.020,  0.042],
    [-0.031, -0.070, -0.110,  0.125,  0.050, -0.060],
    [-0.027, -0.050, -0.020,  0.050,  0.065, -0.020],
    [ 0.010,  0.020,  0.042, -0.060, -0.020,  0.080],
])

# Soluciones de referencia del libro (para calcular error)
REFERENCIA = {
    'P1': {'rendimiento': 0.6920, 'riesgo': 0.045480,
           'cartera': np.array([0.00, 0.00, 0.40, 0.40, 0.20, 0.00])},
    'P2': {'rendimiento': 0.3590, 'riesgo': 0.001360,
           'cartera': np.array([0.318, 0.199, 0.000, 0.168, 0.209, 0.106])},
    'P3': {'rendimiento': None,   'riesgo': 0.002000,
           'cartera': None},
}

# ============================================================
#  CARPETA DE SALIDA
# ============================================================
TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR     = f"./data/bitacoras_{TIMESTAMP}"
os.makedirs(OUT_DIR, exist_ok=True)

LOG_GLOBAL  = os.path.join(OUT_DIR, "log_global.txt")
CSV_P1      = os.path.join(OUT_DIR, "resultados_P1.csv")
CSV_P2      = os.path.join(OUT_DIR, "resultados_P2.csv")
CSV_P3      = os.path.join(OUT_DIR, "resultados_P3.csv")
CSV_DETALLE = os.path.join(OUT_DIR, "detalle_generaciones.csv")
CSV_RESUMEN = os.path.join(OUT_DIR, "resumen_parametros.csv")

# ============================================================
#  UTILIDADES DE CARTERA
# ============================================================
def rendimiento_esperado(x):
    return float(np.dot(RENDIMIENTOS, x))

def riesgo(x):
    return float(x @ COV @ x)

def normalize(x):
    x = np.clip(x, 0.0, 0.40)
    s = np.sum(x)
    return x / s if s > 1e-12 else np.ones(N_ACCIONES) / N_ACCIONES

def error_vs_referencia(x, problema):
    ref = REFERENCIA[problema]
    err_ret  = abs(rendimiento_esperado(x) - ref['rendimiento']) if ref['rendimiento'] else None
    err_risk = abs(riesgo(x) - ref['riesgo'])
    return err_ret, err_risk

# ============================================================
#  RESTRICCIONES Y FITNESS
# ============================================================
def penalizacion_comun(x, pen_coef):
    pen  = abs(np.sum(x) - 1.0) * pen_coef
    pen += np.sum(np.maximum(0.0,  x - 0.40)) * pen_coef
    pen += np.sum(np.maximum(0.0, -x))         * pen_coef
    return pen

def fitness_p1(x, pen_coef):
    return -rendimiento_esperado(x) + penalizacion_comun(x, pen_coef)

def fitness_p2(x, pen_coef):
    pen  = penalizacion_comun(x, pen_coef)
    pen += max(0.0, 0.35 - rendimiento_esperado(x)) * pen_coef
    return riesgo(x) + pen

def fitness_p3(x, pen_coef):
    pen  = penalizacion_comun(x, pen_coef)
    pen += max(0.0, riesgo(x) - 0.002) * pen_coef
    return -rendimiento_esperado(x) + pen

FITNESS_FNS = {'P1': fitness_p1, 'P2': fitness_p2, 'P3': fitness_p3}

LI = np.zeros(N_ACCIONES)
LS = np.ones(N_ACCIONES)

# ============================================================
#  OPERADORES GENÉTICOS
# ============================================================
def get_initial_population(tam):
    pop = []
    for _ in range(tam):
        x = rng.dirichlet(np.ones(N_ACCIONES))
        pop.append(normalize(x))
    return pop

def tournament_selection(pop, fits, t_size):
    selected = []
    n = len(pop)
    for _ in range(n):
        idx  = rng.choice(n, size=t_size, replace=False)
        best = idx[np.argmin([fits[i] for i in idx])]
        selected.append(pop[best].copy())
    return selected

def crossover_sbx(p1, p2, prob_cruce, nc):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() < prob_cruce:
        for j in range(N_ACCIONES):
            P1, P2 = p1[j], p2[j]
            if abs(P1 - P2) < 1e-10:
                continue
            if P1 > P2:
                P1, P2 = P2, P1
            beta  = 1 + (2 / (P2 - P1 + 1e-12)) * min(P1 - LI[j], LS[j] - P2)
            beta  = max(beta, 1.0)
            alpha = 2 - abs(beta) ** -(nc + 1)
            u = rng.random()
            if u < 1.0 / alpha:
                bq = (u * alpha) ** (1.0 / (nc + 1))
            else:
                bq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (nc + 1))
            c1[j] = np.clip(0.5*((P1+P2) - bq*(P2-P1)), LI[j], LS[j])
            c2[j] = np.clip(0.5*((P1+P2) + bq*(P2-P1)), LI[j], LS[j])
    return c1, c2

def polynomial_mutation(x, prob_mut, nm):
    x = x.copy()
    for j in range(N_ACCIONES):
        if rng.random() < prob_mut:
            P     = x[j]
            delta = min(LS[j]-P, P-LI[j]) / (LS[j]-LI[j] + 1e-12)
            r     = rng.random()
            if r < 0.5:
                dq = (2*r + (1-2*r)*(1-delta)**(nm+1))**(1/(nm+1)) - 1
            else:
                dq = 1 - (2*(1-r) + 2*(r-0.5)*(1-delta)**(nm+1))**(1/(nm+1))
            x[j] = np.clip(P + dq*(LS[j]-LI[j]), LI[j], LS[j])
    return x

# ============================================================
#  NÚCLEO DEL AG (un solo run)
# ============================================================
def run_single(problema, params, track_generations=False):
    """
    Ejecuta un run del AG con los parámetros dados.
    Retorna dict con resultados y (opcionalmente) historial por generación.
    """
    tam      = params['tam_poblacion']
    n_gen    = params['n_generaciones']
    nc       = params['nc']
    nm       = params['nm']
    pc       = params['prob_cruce']
    pm       = params['prob_mutacion']
    t_size   = params['torneo']
    pen      = params['penalizacion']
    fit_fn   = FITNESS_FNS[problema]

    pop      = get_initial_population(tam)
    best_x   = None
    best_fit = np.inf
    history  = []   # (gen, best_fit, best_ret, best_risk)

    for gen in range(n_gen):
        fits = [fit_fn(x, pen) for x in pop]

        # Elitismo
        idx_best = int(np.argmin(fits))
        if fits[idx_best] < best_fit:
            best_fit = fits[idx_best]
            best_x   = pop[idx_best].copy()

        # Selección → cruce → mutación
        parents  = tournament_selection(pop, fits, t_size)
        new_pop  = []
        for i in range(0, tam, 2):
            c1, c2 = crossover_sbx(parents[i], parents[i+1], pc, nc)
            c1 = polynomial_mutation(c1, pm, nm)
            c2 = polynomial_mutation(c2, pm, nm)
            new_pop.append(normalize(c1))
            new_pop.append(normalize(c2))

        # Inyectar élite
        new_pop[rng.integers(0, tam)] = normalize(best_x)
        pop = new_pop

        if track_generations and (gen % 50 == 0 or gen == n_gen - 1):
            bx = normalize(best_x)
            history.append((gen, best_fit, rendimiento_esperado(bx), riesgo(bx)))

    # Evaluación final
    fits     = [fit_fn(x, pen) for x in pop]
    idx_best = int(np.argmin(fits))
    if fits[idx_best] < best_fit:
        best_fit = fits[idx_best]
        best_x   = pop[idx_best].copy()

    best_x = normalize(best_x)
    return {
        'x':        best_x,
        'fitness':  best_fit,
        'ret':      rendimiento_esperado(best_x),
        'risk':     riesgo(best_x),
        'history':  history,
    }

# ============================================================
#  LOGGER GLOBAL (TXT)
# ============================================================
def log(msg, also_print=True):
    with open(LOG_GLOBAL, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
    if also_print:
        print(msg)

# ============================================================
#  CABECERAS CSV
# ============================================================
def init_csv(path, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

def append_csv(path, row):
    fieldnames = list(row.keys())
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)

FIELDS_PROB = [
    'experimento', 'problema', 'run',
    'tam_poblacion', 'n_generaciones', 'nc', 'nm',
    'prob_cruce', 'prob_mutacion', 'torneo', 'penalizacion',
    'BB', 'LOP', 'ILI', 'HEAL', 'QUI', 'AUA',
    'rendimiento', 'riesgo', 'fitness',
    'err_rendimiento', 'err_riesgo',
    'tiempo_seg'
]

FIELDS_DETALLE = [
    'experimento', 'problema', 'run', 'generacion',
    'best_fitness', 'rendimiento', 'riesgo'
]

FIELDS_RESUMEN = [
    'experimento', 'problema',
    'tam_poblacion', 'n_generaciones', 'nc', 'nm',
    'prob_cruce', 'prob_mutacion', 'torneo', 'penalizacion',
    'n_runs',
    'mejor_rendimiento', 'mejor_riesgo', 'mejor_fitness',
    'media_rendimiento', 'std_rendimiento',
    'media_riesgo',      'std_riesgo',
    'media_fitness',     'std_fitness',
    'err_medio_rendimiento', 'err_medio_riesgo',
    'tiempo_total_seg'
]

# ============================================================
#  EXPERIMENTO: múltiples runs con un set de parámetros
# ============================================================
exp_counter = [0]

def experimento(problema, params, n_runs, track_gens=False):
    exp_counter[0] += 1
    exp_id   = exp_counter[0]
    csv_path = {'P1': CSV_P1, 'P2': CSV_P2, 'P3': CSV_P3}[problema]

    rets, risks, fits = [], [], []
    err_rets, err_risks = [], []
    t0 = time.time()

    for run in range(n_runs):
        r = run_single(problema, params, track_generations=(track_gens and run == 0))

        # Guardar fila detallada por run
        row = {'experimento': exp_id, 'problema': problema, 'run': run+1}
        row.update({k: params[k] for k in
                    ['tam_poblacion','n_generaciones','nc','nm',
                     'prob_cruce','prob_mutacion','torneo','penalizacion']})
        for i, acc in enumerate(ACCIONES):
            row[acc] = round(r['x'][i], 6)
        row['rendimiento'] = round(r['ret'],  6)
        row['riesgo']      = round(r['risk'], 6)
        row['fitness']     = round(r['fitness'], 6)
        er, erisk = error_vs_referencia(r['x'], problema)
        row['err_rendimiento'] = round(er,    6) if er    is not None else 'N/A'
        row['err_riesgo']      = round(erisk, 6) if erisk is not None else 'N/A'
        row['tiempo_seg']      = 0  # se rellena abajo
        append_csv(csv_path, row)

        # Historial de generaciones (solo run 0 cuando se pide)
        for h in r['history']:
            gen, bf, br, brisk = h
            append_csv(CSV_DETALLE, {
                'experimento': exp_id, 'problema': problema,
                'run': run+1, 'generacion': gen,
                'best_fitness': round(bf, 8),
                'rendimiento': round(br, 6),
                'riesgo': round(brisk, 6),
            })

        rets.append(r['ret'])
        risks.append(r['risk'])
        fits.append(r['fitness'])
        er_val   = er    if er    is not None else np.nan
        erisk_val= erisk if erisk is not None else np.nan
        err_rets.append(er_val)
        err_risks.append(erisk_val)

    elapsed = time.time() - t0

    # Resumen del experimento
    best_idx = int(np.argmin(fits))
    resumen = {
        'experimento':    exp_id,
        'problema':       problema,
        'tam_poblacion':  params['tam_poblacion'],
        'n_generaciones': params['n_generaciones'],
        'nc':             params['nc'],
        'nm':             params['nm'],
        'prob_cruce':     params['prob_cruce'],
        'prob_mutacion':  params['prob_mutacion'],
        'torneo':         params['torneo'],
        'penalizacion':   params['penalizacion'],
        'n_runs':         n_runs,
        'mejor_rendimiento': round(rets[best_idx],  6),
        'mejor_riesgo':      round(risks[best_idx], 6),
        'mejor_fitness':     round(fits[best_idx],  6),
        'media_rendimiento': round(np.mean(rets),   6),
        'std_rendimiento':   round(np.std(rets),    6),
        'media_riesgo':      round(np.mean(risks),  6),
        'std_riesgo':        round(np.std(risks),   6),
        'media_fitness':     round(np.mean(fits),   6),
        'std_fitness':       round(np.std(fits),    6),
        'err_medio_rendimiento': round(np.nanmean(err_rets),  6),
        'err_medio_riesgo':      round(np.nanmean(err_risks), 6),
        'tiempo_total_seg':  round(elapsed, 2),
    }
    append_csv(CSV_RESUMEN, resumen)
    return resumen, rets, risks, fits

# ============================================================
#  BARRIDO DE PARÁMETROS
# ============================================================
GRID = {
    'tam_poblacion':  [100, 200, 400],
    'n_generaciones': [300, 500, 800],
    'nc':             [5, 15, 30],
    'nm':             [10, 20, 40],
    'prob_cruce':     [0.7, 0.9],
    'prob_mutacion':  [0.02, 0.05, 0.10],
    'torneo':         [2, 3, 5],
    'penalizacion':   [1e4, 1e5, 1e6],
}

# Parámetros base (referencia)
BASE = {
    'tam_poblacion':  200,
    'n_generaciones': 500,
    'nc':             15,
    'nm':             20,
    'prob_cruce':     0.9,
    'prob_mutacion':  0.05,
    'torneo':         3,
    'penalizacion':   1e5,
}

# ============================================================
#  MAIN
# ============================================================
if __name__ == '__main__':

    # Inicializar archivos
    init_csv(CSV_P1,      FIELDS_PROB)
    init_csv(CSV_P2,      FIELDS_PROB)
    init_csv(CSV_P3,      FIELDS_PROB)
    init_csv(CSV_DETALLE, FIELDS_DETALLE)
    init_csv(CSV_RESUMEN, FIELDS_RESUMEN)

    inicio_total = time.time()
    log(f"{'='*70}")
    log(f"  PRUEBAS EXHAUSTIVAS — AG SELECCIÓN DE CARTERA")
    log(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Carpeta de salida: {OUT_DIR}")
    log(f"{'='*70}")

    PROBLEMAS = ['P1', 'P2', 'P3']

    # ----------------------------------------------------------
    # FASE 1: Configuración base con muchos runs
    # ----------------------------------------------------------
    N_RUNS_BASE = 30
    log(f"\n{'─'*70}")
    log(f"  FASE 1 — Configuración base, {N_RUNS_BASE} runs por problema")
    log(f"{'─'*70}")

    mejores_global = {}

    for prob in PROBLEMAS:
        log(f"\n  [{prob}] Corriendo {N_RUNS_BASE} runs con parámetros base...")
        res, rets, risks, fits = experimento(prob, BASE, N_RUNS_BASE, track_gens=True)

        best_idx = int(np.argmin(fits))
        log(f"  [{prob}] Mejor rendimiento : {rets[best_idx]*100:.4f}%")
        log(f"  [{prob}] Mejor riesgo      : {risks[best_idx]:.6f}")
        log(f"  [{prob}] Media rendimiento : {np.mean(rets)*100:.4f}% ± {np.std(rets)*100:.4f}%")
        log(f"  [{prob}] Media riesgo      : {np.mean(risks):.6f} ± {np.std(risks):.6f}")
        log(f"  [{prob}] Tiempo            : {res['tiempo_total_seg']:.1f} s")
        mejores_global[prob] = {'rets': rets, 'risks': risks, 'fits': fits}

    # ----------------------------------------------------------
    # FASE 2: Barrido univariante (one-at-a-time)
    # ----------------------------------------------------------
    N_RUNS_SWEEP = 15
    log(f"\n{'─'*70}")
    log(f"  FASE 2 — Barrido univariante (OAT), {N_RUNS_SWEEP} runs c/u")
    log(f"{'─'*70}")

    param_results = {prob: {} for prob in PROBLEMAS}

    for param_name, values in GRID.items():
        log(f"\n  Parámetro: {param_name}  valores: {values}")
        for val in values:
            params = BASE.copy()
            params[param_name] = val
            for prob in PROBLEMAS:
                res, rets, risks, fits = experimento(prob, params, N_RUNS_SWEEP)
                best_idx = int(np.argmin(fits))
                tag = f"{param_name}={val}"
                if tag not in param_results[prob]:
                    param_results[prob][tag] = []
                param_results[prob][tag].append({
                    'ret':  rets[best_idx],
                    'risk': risks[best_idx],
                    'fit':  fits[best_idx],
                })
                log(f"    [{prob}] {param_name}={val:<8}  "
                    f"ret={rets[best_idx]*100:.3f}%  "
                    f"risk={risks[best_idx]:.6f}  "
                    f"t={res['tiempo_total_seg']:.1f}s")

    # ----------------------------------------------------------
    # FASE 3: Combinaciones prometedoras
    # ----------------------------------------------------------
    N_RUNS_COMBO = 20
    log(f"\n{'─'*70}")
    log(f"  FASE 3 — Combinaciones prometedoras, {N_RUNS_COMBO} runs c/u")
    log(f"{'─'*70}")

    COMBOS = [
        # Exploración alta (más diversidad)
        {'tam_poblacion': 400, 'n_generaciones': 800, 'nc': 5,
         'nm': 10, 'prob_cruce': 0.9, 'prob_mutacion': 0.10,
         'torneo': 2, 'penalizacion': 1e5},
        # Exploración media
        {'tam_poblacion': 200, 'n_generaciones': 800, 'nc': 15,
         'nm': 20, 'prob_cruce': 0.9, 'prob_mutacion': 0.05,
         'torneo': 3, 'penalizacion': 1e5},
        # Explotación fuerte (poca diversidad)
        {'tam_poblacion': 400, 'n_generaciones': 800, 'nc': 30,
         'nm': 40, 'prob_cruce': 0.7, 'prob_mutacion': 0.02,
         'torneo': 5, 'penalizacion': 1e5},
        # Penalización alta + exploración
        {'tam_poblacion': 200, 'n_generaciones': 500, 'nc': 10,
         'nm': 15, 'prob_cruce': 0.9, 'prob_mutacion': 0.08,
         'torneo': 2, 'penalizacion': 1e6},
        # Torneo grande + penalización alta
        {'tam_poblacion': 400, 'n_generaciones': 500, 'nc': 20,
         'nm': 20, 'prob_cruce': 0.9, 'prob_mutacion': 0.05,
         'torneo': 5, 'penalizacion': 1e6},
        # Población gigante
        {'tam_poblacion': 400, 'n_generaciones': 300, 'nc': 15,
         'nm': 20, 'prob_cruce': 0.9, 'prob_mutacion': 0.05,
         'torneo': 3, 'penalizacion': 1e5},
    ]

    for i, combo in enumerate(COMBOS, 1):
        log(f"\n  Combo {i}: tam={combo['tam_poblacion']} gen={combo['n_generaciones']} "
            f"nc={combo['nc']} nm={combo['nm']} pc={combo['prob_cruce']} "
            f"pm={combo['prob_mutacion']} t={combo['torneo']} pen={combo['penalizacion']:.0e}")
        for prob in PROBLEMAS:
            res, rets, risks, fits = experimento(prob, combo, N_RUNS_COMBO)
            best_idx = int(np.argmin(fits))
            log(f"    [{prob}]  ret={rets[best_idx]*100:.4f}%  "
                f"risk={risks[best_idx]:.6f}  "
                f"t={res['tiempo_total_seg']:.1f}s")

    # ----------------------------------------------------------
    # FASE 4: Mejor configuración encontrada → runs masivos
    # ----------------------------------------------------------
    N_RUNS_FINAL = 50
    BEST_PARAMS = {
        'tam_poblacion':  400,
        'n_generaciones': 800,
        'nc':             15,
        'nm':             20,
        'prob_cruce':     0.9,
        'prob_mutacion':  0.05,
        'torneo':         3,
        'penalizacion':   1e5,
    }

    log(f"\n{'─'*70}")
    log(f"  FASE 4 — Configuración óptima, {N_RUNS_FINAL} runs por problema")
    log(f"  Parámetros: {BEST_PARAMS}")
    log(f"{'─'*70}")

    resultados_finales = {}
    for prob in PROBLEMAS:
        res, rets, risks, fits = experimento(prob, BEST_PARAMS, N_RUNS_FINAL, track_gens=True)
        best_idx = int(np.argmin(fits))
        resultados_finales[prob] = {
            'rets': rets, 'risks': risks, 'fits': fits,
            'best_idx': best_idx
        }
        log(f"\n  [{prob}] RESULTADO FINAL ({N_RUNS_FINAL} runs):")
        log(f"    Mejor rendimiento : {rets[best_idx]*100:.4f}%")
        log(f"    Mejor riesgo      : {risks[best_idx]:.6f}")
        log(f"    Media rendimiento : {np.mean(rets)*100:.4f}% ± {np.std(rets)*100:.4f}%")
        log(f"    Media riesgo      : {np.mean(risks):.6f} ± {np.std(risks):.6f}")
        log(f"    Tiempo total      : {res['tiempo_total_seg']:.1f} s")

    # ----------------------------------------------------------
    # REPORTE FINAL
    # ----------------------------------------------------------
    elapsed_total = time.time() - inicio_total
    log(f"\n{'='*70}")
    log(f"  REPORTE FINAL — MEJORES SOLUCIONES ENCONTRADAS")
    log(f"{'='*70}")

    REFS_PRINT = {
        'P1': ('Maximizar rendimiento (riesgo ignorado, xi<=40%)',
               'Rendimiento', 0.6920, None),
        'P2': ('Minimizar riesgo (rendimiento>=35%, xi<=40%)',
               'Riesgo',     None,   0.001360),
        'P3': ('Maximizar rendimiento (riesgo<=0.002, xi<=40%)',
               'Rendimiento', None,  0.002000),
    }

    for prob, (desc, _, ref_ret, ref_risk) in REFS_PRINT.items():
        rf = resultados_finales[prob]
        bi = rf['best_idx']
        log(f"\n  {prob}: {desc}")
        log(f"    Mejor rendimiento  : {rf['rets'][bi]*100:.4f}%"
            + (f"  (ref: {ref_ret*100:.2f}%)" if ref_ret else ""))
        log(f"    Mejor riesgo       : {rf['risks'][bi]:.6f}"
            + (f"  (ref: {ref_risk:.6f})" if ref_risk else ""))
        log(f"    Media rendimiento  : {np.mean(rf['rets'])*100:.4f}% ± {np.std(rf['rets'])*100:.4f}%")
        log(f"    Media riesgo       : {np.mean(rf['risks']):.6f} ± {np.std(rf['risks']):.6f}")
        log(f"    Consistencia (std fitness): {np.std(rf['fits']):.6f}")

    log(f"\n{'─'*70}")
    log(f"  Total de experimentos registrados : {exp_counter[0]}")
    log(f"  Tiempo total de ejecución         : {elapsed_total/60:.1f} min ({elapsed_total:.0f} s)")
    log(f"  Archivos generados en: {OUT_DIR}")
    log(f"    - log_global.txt          (bitácora completa)")
    log(f"    - resultados_P1/P2/P3.csv (detalle por run y problema)")
    log(f"    - detalle_generaciones.csv (evolución por generación)")
    log(f"    - resumen_parametros.csv  (resumen por experimento)")
    log(f"{'='*70}")
