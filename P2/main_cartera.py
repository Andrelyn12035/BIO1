import numpy as np

rng = np.random.default_rng()

TAM_POBLACION    = 200
TOT_GENERACIONES = 500
NC = 15  #Entre mas grande los hijos seran menos parecidos a los padres, entre mas pequeño los hijos seran mas parecidos a los padres. influye en la diversidad de la poblacion, entre mas grande mas diversidad, entre mas pequeño menos diversidad
NM = 20
PROB_MUTACION = 0.05
PROB_CRUCE    = 0.9

TOURNAMET_SIZE = 3

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
# PROBLEMA A RESOLVER  (cambiar para alternar entre P1, P2, P3)
# ============================================================
#   'P1' -> Maximizar rendimiento, ignorar riesgo, xi <= 40%
#   'P2' -> Minimizar riesgo, rendimiento >= 35%,  xi <= 40%
#   'P3' -> Maximizar rendimiento, riesgo <= 0.002, xi <= 40%
PROBLEMA = 'P2'


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
        ## riesgo (varianza) <= 0.002
        pen += max(0.0, riesgo_cartera(variables) - 0.002) * PENALIZACION

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
# NORMALIZACION AL SIMPLEX
# ============================================================
def normalize(variables):
    ## Proyecta el vector al simplex factible: xi in [0, 0.40], suma = 1
    variables = np.clip(variables, 0.0, 0.40)
    s = np.sum(variables)
    if s < 1e-12:
        return np.ones(N_ACCIONES) / N_ACCIONES
    return variables / s


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
def add_data_to_csv(run_number, population):
    best_subject  = get_best_solution(population)
    worse_subject = max(population, key=lambda subj: objective_function(subj['variables']))

    fitness_values  = [objective_function(subj['variables']) for subj in population]
    median_fitness  = np.median(fitness_values)
    std_fitness     = np.std(fitness_values)

    with open('results.csv', 'a') as f:
        if run_number == 0:
            header = "Prueba, " + ", ".join(f"Mejor {a}" for a in ACCIONES)
            header += ", Mejor Rendimiento, Mejor Riesgo, Mejor Aptitud"
            header += ", " + ", ".join(f"Peor {a}" for a in ACCIONES)
            header += ", Peor Rendimiento, Peor Riesgo, Peor Aptitud"
            header += ", Mediana, Desviacion Estandar\n"
            f.write(header)

        bv = best_subject['variables']
        wv = worse_subject['variables']
        row = f"{run_number}"
        row += ", " + ", ".join(f"{bv[i]:.6f}" for i in range(N_ACCIONES))
        row += f", {rendimiento_esperado(bv):.6f}, {riesgo_cartera(bv):.6f}, {objective_function(bv):.6f}"
        row += ", " + ", ".join(f"{wv[i]:.6f}" for i in range(N_ACCIONES))
        row += f", {rendimiento_esperado(wv):.6f}, {riesgo_cartera(wv):.6f}, {objective_function(wv):.6f}"
        row += f", {median_fitness:.6f}, {std_fitness:.6f}\n"
        f.write(row)


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
DESCRIPCIONES = {
    'P1': 'Maximizar rendimiento ignorando riesgo (xi <= 40%)',
    'P2': 'Minimizar riesgo con rendimiento >= 35% (xi <= 40%)',
    'P3': 'Maximizar rendimiento con riesgo <= 0.002 (xi <= 40%)',
}
REFERENCIAS = {
    'P1': {'rendimiento': 69.20, 'riesgo': 0.045480},
    'P2': {'rendimiento': 35.90, 'riesgo': 0.001360},
    'P3': {'rendimiento': None,  'riesgo': 0.002000},
}

print(f"\n{'='*58}")
print(f"  ALGORITMO GENETICO — SELECCION DE CARTERA DE INVERSION")
print(f"{'='*58}")
print(f"  Problema  : {PROBLEMA} — {DESCRIPCIONES[PROBLEMA]}")
print(f"  Poblacion : {TAM_POBLACION}   Generaciones: {TOT_GENERACIONES}")
print(f"  NC={NC}  NM={NM}  PC={PROB_CRUCE}  PM={PROB_MUTACION}  Torneo={TOURNAMET_SIZE}")
print(f"  Penalizacion: {PENALIZACION:.0e}")
print(f"{'='*58}\n")

generations_data = []

for _ in range(10):
    initial_population = get_initial_population()
    generations_data.append({
        'test_number':        _,
        'generations':        [],
        'best_global_solution': None,
        'best_global_fitness':  None
    })

    for generation in range(TOT_GENERACIONES):
        best_subject = min(initial_population, key=lambda subj: objective_function(subj['variables']))

        tournament_matrix = get_tournament_matrix()
        selected_parents  = parent_selection(initial_population, tournament_matrix)

        new_population = []
        for i in range(0, TAM_POBLACION, 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            child1, child2 = crossover_sbx(parent1, parent2)
            new_population.append(polinomial_mutation(child1))
            new_population.append(polinomial_mutation(child2))

        generations_data[-1]['generations'].append(initial_population)
        initial_population = new_population

        # Reemplazar un individuo aleatorio con el mejor de la generacion anterior (elitismo)
        random_index = rng.integers(0, TAM_POBLACION)
        initial_population[random_index] = best_subject

    best_global_solution = get_best_solution(initial_population)
    best_global_fitness  = objective_function(best_global_solution['variables'])
    generations_data[-1]['best_global_solution'] = best_global_solution
    generations_data[-1]['best_global_fitness']  = best_global_fitness

    add_data_to_csv(_, initial_population)
    show_best_solution(best_global_solution, _)


# ============================================================
# MEJOR SOLUCION GLOBAL ENTRE TODOS LOS RUNS
# ============================================================
run_index   = None
best_subject = None
for test_run in generations_data:
    for generation in test_run['generations']:
        for subject in generation:
            if best_subject is None or objective_function(subject['variables']) < objective_function(best_subject['variables']):
                best_subject = subject
                run_index    = test_run['test_number']

ref = REFERENCIAS[PROBLEMA]
print(f"\n{'='*58}")
print(f"  MEJOR SOLUCION GLOBAL  (Run {run_index + 1})")
print(f"{'='*58}")
v = best_subject['variables']
print(f"  {'Accion':<6} {'Fraccion':>10}  {'%':>7}")
print(f"  {'------':<6} {'--------':>10}  {'---':>7}")
for i, acc in enumerate(ACCIONES):
    print(f"  {acc:<6} {v[i]:>10.4f}  {v[i]*100:>6.2f}%")
print(f"  {'------':<6} {'--------':>10}  {'---':>7}")
print(f"  {'TOTAL':<6} {np.sum(v):>10.4f}  {np.sum(v)*100:>6.2f}%")
print(f"\n  Rendimiento esperado : {rendimiento_esperado(v)*100:.4f}%", end="")
if ref['rendimiento']:
    print(f"  (referencia: {ref['rendimiento']:.2f}%)", end="")
print()
print(f"  Riesgo (varianza)    : {riesgo_cartera(v):.6f}", end="")
if ref['riesgo']:
    print(f"  (referencia: {ref['riesgo']:.6f})", end="")
print()
print(f"  Aptitud              : {objective_function(v):.6f}")
print(f"\n  Resultados guardados en: results.csv")
print(f"{'='*58}\n")
