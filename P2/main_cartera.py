import numpy as np

rng = np.random.default_rng()

TAM_POBLACION    = 300
TOT_GENERACIONES = 1000
NC = 15  #Entre mas grande los hijos seran menos parecidos a los padres, entre mas pequeño los hijos seran mas parecidos a los padres. influye en la diversidad de la poblacion, entre mas grande mas diversidad, entre mas pequeño menos diversidad
NM = 20
PROB_MUTACION = 0.05
PROB_CRUCE    = 0.9

TOURNAMET_SIZE = 3

N_ELITES = 5  # individuos elite que sobreviven intactos cada generacion

PENALIZACION = 100000

N_ACCIONES = 6
ACCIONES   = ['BB', 'LOP', 'ILI', 'HEAL', 'QUI', 'AUA']

# Rendimientos esperados de cada accion
RENDIMIENTOS = np.array([0.20, 0.42, 1.00, 0.50, 0.46, 0.30])

# Matriz de covarianzas (varianzas en la diagonal)
COV = np.array([
    [ 0.032,  0.005,  0.030, -0.031, -0.027,  0.010],
    [ 0.005,  0.100,  0.085, -0.070, -0.050,  0.020],
    [ 0.030,  0.085,  0.333, -0.110, -0.020,  0.042],
    [-0.031, -0.070, -0.110,  0.125,  0.050, -0.060],
    [-0.027, -0.050, -0.020,  0.050,  0.065, -0.020],
    [ 0.010,  0.020,  0.042, -0.060, -0.020,  0.080],
])

# Limites de cada variable (fraccion invertida por accion: 0% a 100%)
LI = np.zeros(N_ACCIONES)
LS = np.ones(N_ACCIONES)


# ============================================================
# PROBLEMA A RESOLVER  (se itera sobre los tres en el loop principal)
# ============================================================
#   'P1' -> Maximizar rendimiento, ignorar riesgo, xi <= 40%
#   'P2' -> Minimizar riesgo, rendimiento >= 35%,  xi <= 40%
#   'P3' -> Maximizar rendimiento, riesgo <= LIMITE_RIESGO_P3, xi <= 40%
PROBLEMA          = 'P1'   # se sobreescribe en el loop
LIMITE_RIESGO_P3  = 0.02   # se sobreescribe en el loop para P3


# ============================================================
# FUNCIONES DE CARTERA
# ============================================================
def rendimiento_esperado(variables):
    return float(np.dot(RENDIMIENTOS, variables))

def riesgo_cartera(variables):
    return float(variables @ COV @ variables)


# ============================================================
# FUNCION DE RESTRICCIONES  (penalizacion exterior)
# ============================================================
def restriction_function(variables):
    ## Restriccion comun a los tres problemas:
    ##   suma de fracciones = 1  (invertir todo el capital)
    ##   xi <= 0.40              (no mas del 40% en una sola accion)
    ##   xi >= 0                 (no ventas en corto)
    pen = abs(np.sum(variables) - 1.0) * PENALIZACION
    for xi in variables:
        pen += max(0.0, xi - 0.40) * PENALIZACION
        pen += max(0.0, -xi)       * PENALIZACION

    ## Restricciones especificas por problema
    if PROBLEMA == 'P2':
        ## rendimiento esperado >= 35%
        pen += max(0.0, 0.35 - rendimiento_esperado(variables)) * PENALIZACION
    if PROBLEMA == 'P3':
        pen += max(0.0, riesgo_cartera(variables) - LIMITE_RIESGO_P3) * PENALIZACION

    return pen


# ============================================================
# FUNCION OBJETIVO
# ============================================================
def objective_function(variables):
    ## P1: maximizar rendimiento  -> minimizar el negativo
    ## P2: minimizar riesgo
    ## P3: maximizar rendimiento con restriccion de riesgo
    if PROBLEMA == 'P1' or PROBLEMA == 'P3':
        obj = -rendimiento_esperado(variables)
    else:  # P2
        obj = riesgo_cartera(variables)

    return obj + restriction_function(variables)


# ============================================================
# NORMALIZACION
# ============================================================
def normalize(variables):
    ## factible: xi in [0, 0.40], suma = 1
    ## Se itera clip→rescale porque dividir por una suma < 1 puede empujar
    ## valores que estaban en 0.40 por encima del límite.
    variables = np.clip(variables, 0.0, 0.40)
    for _ in range(50):
        s = np.sum(variables)
        if s < 1e-12:
            return np.ones(N_ACCIONES) / N_ACCIONES
        variables = variables / s
        clipped = np.clip(variables, 0.0, 0.40)
        if np.allclose(variables, clipped):
            break
        variables = clipped
    return variables


# ============================================================
# POBLACION INICIAL
# ============================================================
def get_initial_population():
    population = []
    for _ in range(TAM_POBLACION):
        subject = {}
        ## Dirichlet genera vectores que suman 1 naturalmente
        variables = rng.dirichlet(np.ones(N_ACCIONES))
        subject['variables'] = normalize(variables)
        population.append(subject)
    return population


# ============================================================
# SELECCION POR TORNEO
# ============================================================
def get_tournament_matrix():
    tournament_matrix = np.zeros((TAM_POBLACION, TOURNAMET_SIZE), dtype=int)
    for i in range(TAM_POBLACION):
        tournament_matrix[i] = rng.choice(TAM_POBLACION, size=TOURNAMET_SIZE, replace=False)
    return tournament_matrix

def parent_selection(population, tournament_matrix):
    selected_parents = []
    for i in range(TAM_POBLACION):
        tournament_subjects = [population[j] for j in tournament_matrix[i]]
        best_subject = min(tournament_subjects, key=lambda subj: objective_function(subj['variables']))
        selected_parents.append(best_subject)
    return selected_parents


# ============================================================
# CRUCE SBX (Simulated Binary Crossover)
# ============================================================
def crossover_sbx(parent1, parent2):
    child1 = {'variables': np.zeros(N_ACCIONES)}
    child2 = {'variables': np.zeros(N_ACCIONES)}
    if rng.random() < PROB_CRUCE:
        for j in range(N_ACCIONES):
            P1 = parent1['variables'][j]
            P2 = parent2['variables'][j]
            if abs(P1 - P2) < 1e-10:
                child1['variables'][j] = P1
                child2['variables'][j] = P2
                continue
            if P1 > P2:
                P1, P2 = P2, P1
            beta  = 1 + (2 / (P2 - P1 + 1e-12)) * min(P1 - LI[j], LS[j] - P2)
            beta  = max(beta, 1.0)
            alpha = 2 - abs(beta) ** -(NC + 1)
            if rng.random() < 1 / alpha:
                beta_q = (rng.random() * alpha) ** (1 / (NC + 1))
            else:
                beta_q = (1 / (2 - rng.random() * alpha)) ** (1 / (NC + 1))
            child1_j = 0.5 * ((P1 + P2) - beta_q * abs(P1 - P2))
            child2_j = 0.5 * ((P1 + P2) + beta_q * abs(P1 - P2))
            child1['variables'][j] = np.clip(child1_j, LI[j], LS[j])
            child2['variables'][j] = np.clip(child2_j, LI[j], LS[j])
    else:
        child1['variables'] = parent1['variables'].copy()
        child2['variables'] = parent2['variables'].copy()
    return child1, child2


# ============================================================
# MUTACION POLINOMIAL
# ============================================================
def polinomial_mutation(subject):
    if rng.random() < PROB_MUTACION:
        for j in range(N_ACCIONES):
            P     = subject['variables'][j]
            r     = rng.random()
            delta = min(LS[j] - P, P - LI[j]) / (LS[j] - LI[j] + 1e-12)
            if r < 0.5:
                delta_q = (2 * r + (1 - 2 * r) * (1 - delta) ** (NM + 1)) ** (1 / (NM + 1)) - 1
            else:
                delta_q = 1 - (2 * (1 - r) + 2 * (r - 0.5) * (1 - delta) ** (NM + 1)) ** (1 / (NM + 1))
            subject['variables'][j] += delta_q * (LS[j] - LI[j])
        subject['variables'] = normalize(subject['variables'])
    return subject


# ============================================================
# MEJOR SOLUCION DE LA POBLACION
# ============================================================
def get_best_solution(population):
    best_subject = min(population, key=lambda subj: objective_function(subj['variables']))
    return best_subject


# ============================================================
# IMPRIMIR POBLACION
# ============================================================
def show_population(population):
    for subject in population:
        aptitud = objective_function(subject['variables'])
        cartera = ', '.join(f"{acc}:{subject['variables'][i]*100:.1f}%" for i, acc in enumerate(ACCIONES))
        print(f"Subject: [{cartera}]  Aptitud: {aptitud:.6f}")


# ============================================================
# GUARDAR RESULTADOS EN CSV
# ============================================================
def add_data_to_csv(run_results, csv_filename):
    """
    run_results: lista de dicts con claves run, variables, rendimiento, riesgo, aptitud.
    Genera dos secciones:
      1) Resultados por ejecucion (pesos, rendimiento, riesgo, aptitud)
      2) Tabla de metricas (Mejor, Media, Peor, Desv. estandar) sobre las aptitudes
    """
    aptitudes = [r['aptitud'] for r in run_results]
    mejor_run = min(run_results, key=lambda r: r['aptitud'])

    with open(csv_filename, 'w') as f:
        # --- Seccion 1: resultados por ejecucion ---
        header = "Ejecucion, " + ", ".join(f"{a} (%)" for a in ACCIONES)
        header += ", Rendimiento (%), Riesgo, Aptitud\n"
        f.write(header)

        for r in run_results:
            v   = r['variables']
            row = f"{r['run'] + 1}"
            row += ", " + ", ".join(f"{v[i]*100:.4f}" for i in range(N_ACCIONES))
            row += f", {r['rendimiento']*100:.4f}, {r['riesgo']:.6f}, {r['aptitud']:.6f}\n"
            f.write(row)

        # --- Seccion 2: tabla de metricas ---
        f.write("\n")
        f.write("Indicador, Valor\n")
        f.write(f"Mejor,          {min(aptitudes):.6f}\n")
        f.write(f"Media,          {np.mean(aptitudes):.6f}\n")
        f.write(f"Peor,           {max(aptitudes):.6f}\n")
        f.write(f"Desv. estandar, {np.std(aptitudes):.6f}\n")

        # --- Seccion 3: solucion optima global ---
        f.write("\n")
        f.write("Solucion optima (mejor ejecucion)\n")
        v = mejor_run['variables']
        f.write("Accion, Fraccion, %\n")
        for i, acc in enumerate(ACCIONES):
            f.write(f"{acc}, {v[i]:.6f}, {v[i]*100:.4f}%\n")
        f.write(f"Rendimiento esperado, {mejor_run['rendimiento']*100:.4f}%\n")
        f.write(f"Riesgo (varianza),    {mejor_run['riesgo']:.6f}\n")


# ============================================================
# IMPRIMIR SOLUCION FINAL
# ============================================================
def show_best_solution(subject, run_number):
    v = subject['variables']
    print(f"\n  Run {run_number + 1:>2} | Generacion final")
    print(f"  {'Accion':<6} {'Fraccion':>10}  {'%':>7}")
    print(f"  {'------':<6} {'--------':>10}  {'---':>7}")
    for i, acc in enumerate(ACCIONES):
        print(f"  {acc:<6} {v[i]:>10.4f}  {v[i]*100:>6.2f}%")
    print(f"  {'------':<6} {'--------':>10}  {'---':>7}")
    print(f"  {'TOTAL':<6} {np.sum(v):>10.4f}  {np.sum(v)*100:>6.2f}%")
    print(f"  Rendimiento esperado : {rendimiento_esperado(v)*100:.4f}%")
    print(f"  Riesgo (varianza)    : {riesgo_cartera(v):.6f}")
    print(f"  Aptitud              : {objective_function(v):.6f}")


# ============================================================
# LOOP PRINCIPAL
# Cada entrada: (PROBLEMA, LIMITE_RIESGO_P3, descripcion, csv)
CONFIGURACIONES = [
    ('P1', None,  'Maximizar rendimiento ignorando riesgo (xi <= 40%)',               'ResultadosSegundaEjecucionP1.csv'),
    ('P2', None,  'Minimizar riesgo con rendimiento >= 35% (xi <= 40%)',              'ResultadosSegundaEjecucionP2.csv'),
    ('P3', 0.02,  'Maximizar rendimiento con riesgo <= 0.02  (xi <= 40%)',            'ResultadosSegundaEjecucionP3_limite002.csv'),
    ('P3', 0.002, 'Maximizar rendimiento con riesgo <= 0.002 (xi <= 40%)',            'ResultadosSegundaEjecucionP3_limite0002.csv'),
]

for PROBLEMA, LIMITE_RIESGO_P3, descripcion, csv_filename in CONFIGURACIONES:
    print(f"\n{'='*58}")
    print(f"  ALGORITMO GENETICO — SELECCION DE CARTERA DE INVERSION")
    print(f"{'='*58}")
    print(f"  Problema  : {PROBLEMA} — {descripcion}")
    print(f"  Poblacion : {TAM_POBLACION}   Generaciones: {TOT_GENERACIONES}")
    print(f"  NC={NC}  NM={NM}  PC={PROB_CRUCE}  PM={PROB_MUTACION}  Torneo={TOURNAMET_SIZE}")
    print(f"  Penalizacion: {PENALIZACION:.0e}")
    print(f"{'='*58}\n")

    run_results = []

    for _ in range(10):
        initial_population = get_initial_population()

        for generation in range(TOT_GENERACIONES):
            elites = sorted(initial_population, key=lambda subj: objective_function(subj['variables']))[:N_ELITES]

            tournament_matrix = get_tournament_matrix()
            selected_parents  = parent_selection(initial_population, tournament_matrix)

            new_population = []
            for i in range(0, TAM_POBLACION, 2):
                parent1 = selected_parents[i]
                parent2 = selected_parents[i + 1]
                child1, child2 = crossover_sbx(parent1, parent2)
                new_population.append(polinomial_mutation(child1))
                new_population.append(polinomial_mutation(child2))

            # Reemplazar los N_ELITES peores con los elites de la generacion anterior
            new_population.sort(key=lambda subj: objective_function(subj['variables']), reverse=True)
            for k in range(N_ELITES):
                new_population[k] = {'variables': elites[k]['variables'].copy()}

            initial_population = new_population

        best_sol = get_best_solution(initial_population)
        run_results.append({
            'run':         _,
            'variables':   best_sol['variables'],
            'rendimiento': rendimiento_esperado(best_sol['variables']),
            'riesgo':      riesgo_cartera(best_sol['variables']),
            'aptitud':     objective_function(best_sol['variables']),
        })
        show_best_solution(best_sol, _)

    add_data_to_csv(run_results, csv_filename)

    # ============================================================
    # MEJOR SOLUCION GLOBAL ENTRE TODOS LOS RUNS DEL PROBLEMA
    # ============================================================
    mejor_run = min(run_results, key=lambda r: r['aptitud'])
    v = mejor_run['variables']

    print(f"\n{'='*58}")
    print(f"  MEJOR SOLUCION GLOBAL {PROBLEMA}  (Run {mejor_run['run'] + 1})")
    print(f"{'='*58}")
    print(f"  {'Accion':<6} {'Fraccion':>10}  {'%':>7}")
    print(f"  {'------':<6} {'--------':>10}  {'---':>7}")
    for i, acc in enumerate(ACCIONES):
        print(f"  {acc:<6} {v[i]:>10.4f}  {v[i]*100:>6.2f}%")
    print(f"  {'------':<6} {'--------':>10}  {'---':>7}")
    print(f"  {'TOTAL':<6} {np.sum(v):>10.4f}  {np.sum(v)*100:>6.2f}%")
    print(f"\n  Rendimiento esperado : {mejor_run['rendimiento']*100:.4f}%")
    print(f"  Riesgo (varianza)    : {mejor_run['riesgo']:.6f}")
    print(f"  Aptitud              : {mejor_run['aptitud']:.6f}")

    aptitudes = [r['aptitud'] for r in run_results]
    print(f"\n  {'Indicador':<20} {'Valor':>12}")
    print(f"  {'Mejor':<20} {min(aptitudes):>12.6f}")
    print(f"  {'Media':<20} {np.mean(aptitudes):>12.6f}")
    print(f"  {'Peor':<20} {max(aptitudes):>12.6f}")
    print(f"  {'Desv. estandar':<20} {np.std(aptitudes):>12.6f}")
    print(f"\n  Resultados guardados en: {csv_filename}")
    print(f"{'='*58}\n")
