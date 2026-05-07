import numpy as np

# ============================================================
#  DATOS DEL PROBLEMA
# ============================================================
rng = np.random.default_rng(42)

ACCIONES = ['BB', 'LOP', 'ILI', 'HEAL', 'QUI', 'AUA']
N_ACCIONES = len(ACCIONES)

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

# ============================================================
#  PARÁMETROS DEL ALGORITMO GENÉTICO
# ============================================================
TAM_POBLACION   = 200
TOT_GENERACIONES = 500
NC              = 15    # Distribución SBX: mayor = más diversidad
NM              = 20    # Distribución mutación polinomial
PROB_CRUCE      = 0.9
PROB_MUTACION   = 0.05
TOURNAMENT_SIZE = 3
PENALIZACION    = 1e5   # Penalización exterior por restricción violada

# Límites de cada gen (fracción de inversión por acción)
LI = np.zeros(N_ACCIONES)   # mínimo 0%
LS = np.ones(N_ACCIONES)    # máximo 100% (restricciones lo acotan)


# ============================================================
#  FUNCIONES DE CARTERA
# ============================================================
def rendimiento_esperado(x):
    return float(np.dot(RENDIMIENTOS, x))

def riesgo(x):
    return float(x @ COV @ x)


# ============================================================
#  RESTRICCIONES COMUNES (aplicadas en todos los problemas)
#  - Suma = 1  (invertir todo el dinero)
#  - xi <= 0.40  (no más del 40% en una sola acción)
#  - xi >= 0    (no ventas en corto)
# ============================================================
def penalizacion_comun(x):
    pen = 0.0
    # |suma - 1| = 0  →  penalizar desvíos
    pen += abs(np.sum(x) - 1.0) * PENALIZACION
    # xi <= 0.40
    for xi in x:
        pen += max(0.0, xi - 0.40) * PENALIZACION
        # xi >= 0  (aunque LI=0 ya lo controla, reforzamos)
        pen += max(0.0, -xi) * PENALIZACION
    return pen


# ============================================================
#  FUNCIONES OBJETIVO POR PROBLEMA
# ============================================================

# --- Problema 1 ---
# Maximizar rendimiento esperado (ignorar riesgo)
# Restricciones: suma=1, xi<=0.40, xi>=0
# Minimizamos el negativo del rendimiento.
def fitness_p1(x):
    obj = -rendimiento_esperado(x)   # minimizar → maximizar rendimiento
    pen = penalizacion_comun(x)
    return obj + pen

# --- Problema 2 ---
# Minimizar riesgo
# Restricciones comunes + rendimiento >= 35%
def fitness_p2(x):
    obj = riesgo(x)
    pen = penalizacion_comun(x)
    pen += max(0.0, 0.35 - rendimiento_esperado(x)) * PENALIZACION
    return obj + pen

# --- Problema 3 ---
# Maximizar rendimiento esperado
# Restricciones comunes + riesgo <= 0.002
def fitness_p3(x):
    obj = -rendimiento_esperado(x)
    pen = penalizacion_comun(x)
    pen += max(0.0, riesgo(x) - 0.002) * PENALIZACION
    return obj + pen


# ============================================================
#  ALGORITMO GENÉTICO (real-coded)
# ============================================================

def get_initial_population():
    """Genera población inicial con vectores normalizados (suma=1)."""
    pop = []
    for _ in range(TAM_POBLACION):
        x = rng.dirichlet(np.ones(N_ACCIONES))   # suma exacta = 1
        pop.append(x)
    return pop

def tournament_selection(pop, fitness_fn):
    selected = []
    fits = [fitness_fn(x) for x in pop]
    for _ in range(TAM_POBLACION):
        idx = rng.choice(TAM_POBLACION, size=TOURNAMENT_SIZE, replace=False)
        best = idx[np.argmin([fits[i] for i in idx])]
        selected.append(pop[best].copy())
    return selected

def crossover_sbx(p1, p2):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() < PROB_CRUCE:
        for j in range(N_ACCIONES):
            P1, P2 = p1[j], p2[j]
            if abs(P1 - P2) < 1e-10:
                continue
            # Aseguramos orden
            if P1 > P2:
                P1, P2 = P2, P1
            beta = 1 + (2 / (P2 - P1 + 1e-12)) * min(P1 - LI[j], LS[j] - P2)
            beta = max(beta, 1.0)
            alpha = 2 - abs(beta) ** -(NC + 1)
            u = rng.random()
            if u < 1.0 / alpha:
                beta_q = (u * alpha) ** (1.0 / (NC + 1))
            else:
                beta_q = (1.0 / (2.0 - u * alpha)) ** (1.0 / (NC + 1))
            c1[j] = np.clip(0.5 * ((P1 + P2) - beta_q * (P2 - P1)), LI[j], LS[j])
            c2[j] = np.clip(0.5 * ((P1 + P2) + beta_q * (P2 - P1)), LI[j], LS[j])
    return c1, c2

def polynomial_mutation(x):
    x = x.copy()
    for j in range(N_ACCIONES):
        if rng.random() < PROB_MUTACION:
            P = x[j]
            delta = min(LS[j] - P, P - LI[j]) / (LS[j] - LI[j] + 1e-12)
            r = rng.random()
            if r < 0.5:
                delta_q = (2*r + (1 - 2*r)*(1 - delta)**(NM+1))**(1/(NM+1)) - 1
            else:
                delta_q = 1 - (2*(1-r) + 2*(r-0.5)*(1-delta)**(NM+1))**(1/(NM+1))
            x[j] = np.clip(P + delta_q * (LS[j] - LI[j]), LI[j], LS[j])
    return x

def normalize(x):
    """Proyecta x al simplex (suma=1, xi>=0)."""
    x = np.clip(x, 0, 0.40)
    s = np.sum(x)
    if s < 1e-12:
        return np.ones(N_ACCIONES) / N_ACCIONES
    return x / s

def run_ga(fitness_fn, problema_nombre, n_runs=5):
    """Ejecuta el AG varias veces y devuelve la mejor solución global."""
    print(f"\n{'='*60}")
    print(f"  {problema_nombre}")
    print(f"{'='*60}")

    global_best_x   = None
    global_best_fit = np.inf

    for run in range(n_runs):
        pop = get_initial_population()
        best_x   = None
        best_fit = np.inf

        for gen in range(TOT_GENERACIONES):
            # Elitismo: guardar el mejor
            fits = [fitness_fn(x) for x in pop]
            local_best_idx = int(np.argmin(fits))
            if fits[local_best_idx] < best_fit:
                best_fit = fits[local_best_idx]
                best_x   = pop[local_best_idx].copy()

            # Selección
            parents = tournament_selection(pop, fitness_fn)

            # Cruce + Mutación
            new_pop = []
            for i in range(0, TAM_POBLACION, 2):
                c1, c2 = crossover_sbx(parents[i], parents[i+1])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                # Normalizar para mantener suma cercana a 1
                new_pop.append(normalize(c1))
                new_pop.append(normalize(c2))

            # Reemplazar un individuo aleatorio con el élite
            new_pop[rng.integers(0, TAM_POBLACION)] = normalize(best_x)
            pop = new_pop

        # Evaluar población final
        fits = [fitness_fn(x) for x in pop]
        local_best_idx = int(np.argmin(fits))
        if fits[local_best_idx] < best_fit:
            best_fit = fits[local_best_idx]
            best_x   = pop[local_best_idx].copy()

        x_norm = normalize(best_x)
        ret = rendimiento_esperado(x_norm)
        risk = riesgo(x_norm)
        print(f"  Run {run+1:>2} | Fitness: {best_fit:>12.6f} | Rendimiento: {ret:.4f} | Riesgo: {risk:.6f}")

        if best_fit < global_best_fit:
            global_best_fit = best_fit
            global_best_x   = x_norm.copy()

    return global_best_x, global_best_fit


def print_solution(x, problema_nombre):
    x = normalize(x)
    print(f"\n{'─'*60}")
    print(f"  SOLUCIÓN ÓPTIMA — {problema_nombre}")
    print(f"{'─'*60}")
    print(f"  {'Acción':<8} {'Fracción':>10} {'%':>8}")
    print(f"  {'------':<8} {'--------':>10} {'---':>8}")
    for i, acc in enumerate(ACCIONES):
        print(f"  {acc:<8} {x[i]:>10.4f} {x[i]*100:>7.2f}%")
    print(f"  {'─'*36}")
    print(f"  {'TOTAL':<8} {np.sum(x):>10.4f} {np.sum(x)*100:>7.2f}%")
    print(f"\n  Rendimiento esperado : {rendimiento_esperado(x)*100:.4f}%")
    print(f"  Riesgo (varianza)    : {riesgo(x):.6f}")
    print(f"{'─'*60}")


# ============================================================
#  MAIN
# ============================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("  ALGORITMO GENÉTICO — SELECCIÓN DE CARTERA DE INVERSIÓN")
    print("  Hillier & Lieberman — Caso 12.1 / 13.1")
    print("="*60)
    print(f"\n  Parámetros del AG:")
    print(f"    Tamaño de población : {TAM_POBLACION}")
    print(f"    Generaciones        : {TOT_GENERACIONES}")
    print(f"    Prob. cruce         : {PROB_CRUCE}")
    print(f"    Prob. mutación      : {PROB_MUTACION}")
    print(f"    Tamaño torneo       : {TOURNAMENT_SIZE}")
    print(f"    Penalización        : {PENALIZACION:.0e}")
    print(f"    Runs por problema   : 5")

    # --- Problema 1 ---
    best_x1, _ = run_ga(fitness_p1, "PROBLEMA 1 — Maximizar rendimiento (riesgo ignorado, xi<=40%)")
    print_solution(best_x1, "PROBLEMA 1")
    print("\n  Solución esperada (manual): 40% ILI, 40% HEAL, 20% QUI")
    print(f"  Rendimiento esperado ref  : 69.20%")
    print(f"  Riesgo ref                : 0.045480")

    # --- Problema 2 ---
    best_x2, _ = run_ga(fitness_p2, "PROBLEMA 2 — Minimizar riesgo con rendimiento >= 35%")
    print_solution(best_x2, "PROBLEMA 2")
    print("\n  Solución esperada (manual): 31.8%BB, 19.9%LOP, 16.8%HEAL, 20.9%QUI, 10.6%AUA")
    print(f"  Rendimiento esperado ref  : 35.90%")
    print(f"  Riesgo ref                : 0.001360")

    # --- Problema 3 ---
    best_x3, _ = run_ga(fitness_p3, "PROBLEMA 3 — Maximizar rendimiento con riesgo <= 0.002")
    print_solution(best_x3, "PROBLEMA 3")
    print("\n  Nivel de riesgo máximo permitido: 0.002000")
    print()
