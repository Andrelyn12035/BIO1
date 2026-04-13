import matplotlib
matplotlib.use('Agg')          # renderizado sin ventana, para guardar directamente a archivo
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from scipy.signal import convolve2d

# ─── Configuración ───────────────────────────────────────────────────────────
MODE     = "sobel"          # "sobel" | "entropia"
TEAM_DIR = "./imagenes/equipo 8/"

TAM_POBLACION    = 50
TOT_GENERACIONES = 40
N_RUNS           = 10          # pruebas independientes por imagen

NC = 10   # índice SBX: mayor → hijos más dispersos respecto a los padres
NM = 20   # índice mutación polinomial: mayor → perturbaciones más pequeñas
TOURNAMET_SIZE = 3

# Probabilidades emparejadas (una configuración por prueba)
PROB_MUTACION = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
PROB_CRUCE    = [0.1,  0.2,  0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Espacio de búsqueda: α (steepness) ∈ [0.1, 10],  Δ (midpoint) ∈ [0, 1]
LI = [0.1, 0.0]
LS = [10.0, 1.0]

rng            = np.random.default_rng()
START_DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR     = "sobel_2" if MODE == "sobel" else "entropia_2"


# ─── Utilidades de imagen ────────────────────────────────────────────────────

def ensure_uint8(image):
    """Normaliza cualquier imagen a uint8 [0-255]."""
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def to_grayscale(image):
    """Convierte a escala de grises con luminosidad estándar (BT.601)."""
    image = ensure_uint8(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:   # descarta canal alfa (RGBA)
        image = image[:, :, :3]
    return np.clip(
        0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2],
        0, 255
    ).astype(np.uint8)


def apply_sigmoid(image, alpha, delta):
    """
    Aplica transformación sigmoide como LUT de 256 entradas.
      S(i) = 255 / (1 + exp(-alpha * (i/255 - delta)))
    α controla la pendiente; Δ controla el punto de inflexión.
    """
    image = ensure_uint8(image)
    if not (np.isfinite(alpha) and np.isfinite(delta)):
        return image                          # parámetros inválidos → devuelve original
    i = np.arange(256) / 255.0
    z = np.clip(-alpha * (i - delta), -60, 60)
    lut = np.clip(np.nan_to_num(255 / (1 + np.exp(z)), nan=128.0), 0, 255).astype(np.uint8)
    return lut[image]


def get_entropy(image):
    """Entropía de Shannon sobre el histograma de intensidades."""
    hist, _ = np.histogram(ensure_uint8(image), bins=256, range=(0, 256))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def get_sobel(image):
    """Magnitud media del gradiente Sobel (detección de bordes)."""
    img = ensure_uint8(image).astype(np.float32)
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1]], dtype=np.float32)
    gx = convolve2d(img, Kx, mode='same', boundary='symm')
    gy = convolve2d(img, Ky, mode='same', boundary='symm')
    return float(np.mean(np.sqrt(gx**2 + gy**2)))


# ─── Función objetivo ────────────────────────────────────────────────────────

def objective_function(subject, image_gray):
    """
    Evalúa un individuo (α, Δ) aplicando la sigmoide y calculando
    la métrica de calidad. Se minimiza el negativo para maximizar la métrica.
    """
    alpha, delta = subject['variables']
    transformed  = apply_sigmoid(image_gray, alpha, delta)
    if transformed.std() < 2:      # imagen colapsada → penalizar
        return 0
    score = get_sobel(transformed) if MODE == "sobel" else get_entropy(transformed)
    return -score


def evaluate_population(population, image_gray):
    """Evalúa toda la población en paralelo y asigna el fitness a cada sujeto."""
    fn = partial(objective_function, image_gray=image_gray)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        fitness_values = list(executor.map(fn, population))
    for subject, fit in zip(population, fitness_values):
        subject['fitness'] = fit
    return population


# ─── Operadores del AG ───────────────────────────────────────────────────────

def get_initial_population():
    """Genera la población inicial con variables aleatorias uniformes."""
    return [
        {'variables': np.array([rng.uniform(LI[0], LS[0]),
                                rng.uniform(LI[1], LS[1])])}
        for _ in range(TAM_POBLACION)
    ]


def tournament_selection(population):
    """Selección por torneo: elige al mejor de TOURNAMET_SIZE candidatos aleatorios."""
    selected = []
    for _ in range(TAM_POBLACION):
        indices = rng.choice(TAM_POBLACION, size=TOURNAMET_SIZE, replace=False)
        best    = min((population[i] for i in indices), key=lambda s: s['fitness'])
        selected.append(best)
    return selected


def crossover_sbx(parent1, parent2, prob_cruce):
    """
    Cruce SBX (Simulated Binary Crossover) con manejo de límites.
    NC controla la dispersión: mayor NC → hijos más cercanos a los padres.
    """
    child1 = {'variables': np.zeros(2)}
    child2 = {'variables': np.zeros(2)}
    if rng.random() < prob_cruce:
        for j in range(2):
            P1 = parent1['variables'][j]
            P2 = parent2['variables'][j]
            p1_safe = P1 if P1 != 0 else 1e-10   # evita división por cero cuando Δ=0
            beta   = 1 + (2 / p1_safe - P2) * min(P1 - LI[j], LS[j] - P2)
            alpha  = 2 - abs(beta) ** -(NC + 1)
            if rng.random() < 1 / alpha:
                beta_q = (rng.random() * alpha) ** (1 / (NC + 1))
            else:
                beta_q = (1 / (2 - rng.random() * alpha)) ** (1 / (NC + 1))
            c1 = 0.5 * ((P1 + P2) - beta_q * abs(P1 - P2))
            c2 = 0.5 * ((P1 + P2) + beta_q * abs(P1 - P2))
            child1['variables'][j] = np.clip(c1, LI[j], LS[j])
            child2['variables'][j] = np.clip(c2, LI[j], LS[j])
    else:
        child1['variables'] = parent1['variables'].copy()
        child2['variables'] = parent2['variables'].copy()
    return child1, child2


def polynomial_mutation(subject, prob_mutacion):
    """
    Mutación polinomial con perturbación acotada.
    NM controla el tamaño del paso: mayor NM → mutaciones más pequeñas.
    """
    if rng.random() < prob_mutacion:
        for j in range(len(subject['variables'])):
            P     = subject['variables'][j]
            r     = rng.random()
            delta = min(LS[j] - P, P - LI[j]) / (LS[j] - LI[j])
            if r < 0.5:
                delta_q = (2*r + (1 - 2*r) * (1 - delta)**(NM + 1))**(1/(NM + 1)) - 1
            else:
                delta_q = 1 - (2*(1-r) + 2*(r - 0.5) * (1 - delta)**(NM + 1))**(1/(NM + 1))
            subject['variables'][j] = np.clip(
                subject['variables'][j] + delta_q * (LS[j] - LI[j]),
                LI[j], LS[j]
            )
    return subject


# ─── Guardado de resultados ──────────────────────────────────────────────────

def save_result_image(subject, image_raw, image_name, tag):
    """Guarda la imagen transformada con los mejores parámetros encontrados."""
    alpha, delta = subject['variables']
    transformed  = apply_sigmoid(image_raw, alpha, delta)
    plt.figure()
    plt.imshow(transformed, cmap='gray' if transformed.ndim == 2 else None)
    plt.axis('off')
    plt.savefig(
        f"{OUTPUT_DIR}/results/{image_name}_{tag}_{START_DATETIME}.png",
        bbox_inches='tight', pad_inches=0
    )
    plt.close()


def append_to_csv(image_name, run_number, population, prob_mutacion, prob_cruce):
    """Agrega una fila con estadísticas de la prueba al CSV de resultados."""
    best  = min(population, key=lambda s: s['fitness'])
    worst = max(population, key=lambda s: s['fitness'])
    fits  = [s['fitness'] for s in population]
    csv_path = f"{OUTPUT_DIR}/results_{image_name}_{START_DATETIME}.csv"
    with open(csv_path, 'a') as f:
        if run_number == 0:
            f.write("Fecha,Prueba,Mejor X,Mejor Y,Mejor solucion,"
                    "Peor X,Peor Y,Peor Solucion,Mediana,Desviacion estandar,"
                    "Probabilidad de mutacion,Probabilidad de cruce\n")
        f.write(
            f"{START_DATETIME},{run_number},"
            f"{best['variables'][0]},{best['variables'][1]},{best['fitness']},"
            f"{worst['variables'][0]},{worst['variables'][1]},{worst['fitness']},"
            f"{np.median(fits)},{np.std(fits)},{prob_mutacion},{prob_cruce}\n"
        )


def save_3d_plot(xyz_points, image_name):
    """Genera y guarda la gráfica 3D de dispersión (α, Δ, fitness)."""
    xs, ys, zs = zip(*xyz_points)
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c='b', marker='o', s=2, alpha=0.4)
    ax.set_xlabel('Alpha (α)')
    ax.set_ylabel('Delta (Δ)')
    ax.set_zlabel('Fitness')
    ax.set_title(f'{image_name} – {MODE}')
    plt.savefig(f"{OUTPUT_DIR}/plots/{image_name}_scatter_{START_DATETIME}.png")
    plt.close()
    with open(f"{OUTPUT_DIR}/plots/{image_name}_data_{START_DATETIME}.csv", 'w') as f:
        f.write("X,Y,Fitness\n")
        for x, y, fit in xyz_points:
            f.write(f"{x},{y},{fit}\n")


# ─── Ciclo principal del AG ──────────────────────────────────────────────────

def run_ga(image_gray, image_raw, image_name):
    """Ejecuta N_RUNS corridas del AG sobre una imagen y guarda todos los resultados."""
    print(f"\n  [{image_name}]")
    all_xyz_points = []

    for run in range(N_RUNS):
        print(f"    Prueba {run + 1}/{N_RUNS}  "
              f"(pm={PROB_MUTACION[run]}, pc={PROB_CRUCE[run]})")
        population = get_initial_population()

        for _ in range(TOT_GENERACIONES):
            # 1) Evaluar
            population  = evaluate_population(population, image_gray)
            best_elite  = min(population, key=lambda s: s['fitness'])

            # 2) Selección → cruce → mutación
            parents = tournament_selection(population)
            new_pop = []
            for i in range(0, TAM_POBLACION, 2):
                c1, c2 = crossover_sbx(parents[i], parents[i+1], PROB_CRUCE[run])
                new_pop.append(polynomial_mutation(c1, PROB_MUTACION[run]))
                new_pop.append(polynomial_mutation(c2, PROB_MUTACION[run]))

            # 3) Elitismo: reemplaza un individuo aleatorio con el mejor anterior
            new_pop[rng.integers(0, TAM_POBLACION)] = best_elite
            population = new_pop

        # Evaluar generación final y recolectar datos
        population = evaluate_population(population, image_gray)
        all_xyz_points.extend(
            (s['variables'][0], s['variables'][1], s['fitness'])
            for s in population
        )

        best = min(population, key=lambda s: s['fitness'])
        print(f"      → α={best['variables'][0]:.4f}  Δ={best['variables'][1]:.4f}  "
              f"fitness={best['fitness']:.4f}")
        append_to_csv(image_name, run, population, PROB_MUTACION[run], PROB_CRUCE[run])

    # Mejor solución global entre todas las pruebas
    best_xyz     = min(all_xyz_points, key=lambda p: p[2])
    best_subject = {'variables': np.array([best_xyz[0], best_xyz[1]]),
                    'fitness':   best_xyz[2]}
    save_result_image(best_subject, image_raw, image_name, "best")
    save_3d_plot(all_xyz_points, image_name)
    print(f"  Mejor global → α={best_xyz[0]:.4f}  Δ={best_xyz[1]:.4f}  "
          f"fitness={best_xyz[2]:.4f}")


# ─── Punto de entrada ────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(f"{OUTPUT_DIR}/plots",   exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/results", exist_ok=True)

    VALID_EXT   = ('.png', '.jpg', '.jpeg')
    image_files = sorted(
        f for f in os.listdir(TEAM_DIR)
        if f.lower().endswith(VALID_EXT)
    )

    if not image_files:
        raise FileNotFoundError(f"No se encontraron imágenes en '{TEAM_DIR}'")

    print(f"Modo      : {MODE.upper()}")
    print(f"Carpeta   : {TEAM_DIR}")
    print(f"Imágenes  : {image_files}")
    print(f"Resultados: {OUTPUT_DIR}/\n")

    for filename in image_files:
        path       = os.path.join(TEAM_DIR, filename)
        image_name = os.path.splitext(filename)[0]
        image_raw  = plt.imread(path)
        image_gray = to_grayscale(image_raw)
        run_ga(image_gray, image_raw, image_name)

    print("\nOptimización completada.")
