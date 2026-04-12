import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import os
import math
from scipy.signal import convolve2d
rng = np.random.default_rng()

TAM_POBLACION = 30
TOT_GENERACIONES = 40
NC = 10 #Entre mas grande los hijos seran menos parecidos a los padres, entre mas pequeño los hijos seran mas parecidos a los padres. influye en la diversidad de la poblacion, entre mas grande mas diversidad, entre mas pequeño menos diversidad
NM = 20
PROB_MUTACION = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
PROB_CRUCE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PRECISION = 8

##considerando 
# α = x
LIMITE_INFERIORX = 0.1
LIMITE_SUPERIORX = 10.0

# Δ = y
LIMITE_INFERIORY = 0.0
LIMITE_SUPERIORY = 1.0


TOURNAMET_SIZE = 3
RESOLUTION = .1


LI = [0.1,0.0]
LS = [10.0,1.0]

IMAGE_DIR = "./imagenes/equipo 7/RSNA_Mammography_1058522855.png"
IMAGE_OBJECT = plt.imread(IMAGE_DIR)

START_DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_uint8(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image = image * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)

def to_grayscale(image):
    image = ensure_uint8(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        gray = (
            0.299 * image[:, :, 0] +
            0.587 * image[:, :, 1] +
            0.114 * image[:, :, 2]
        )
        return np.clip(gray, 0, 255).astype(np.uint8)
    raise ValueError("Formato de imagen no soportado")

IMAGE_GRAY = to_grayscale(IMAGE_OBJECT)


def objective_function(subject):
    x = subject['variables'][0]
    y = subject['variables'][1]
    transformed_image = get_sigmoid_transform(IMAGE_GRAY, x, y)
    # Optional safeguard against collapse
    if transformed_image.std() < 2:
        return 0
    contraste = get_sobel(transformed_image)
    return -contraste

def get_sigmoid_transform(image, x, y):
    image = ensure_uint8(image)
    i = np.arange(256) / 255.0
    z = -x * (i - y)
    z = np.clip(z, -60, 60)

    lut = 255 * (1 / (1 + np.exp(z)))
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return lut[image]

def get_sigmoid_image(image, x, y):
    image = ensure_uint8(image)
    i = np.arange(256) / 255.0
    z = -x * (i - y)
    z = np.clip(z, -60, 60)

    lut = 255 * (1 / (1 + np.exp(z)))
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return lut[image]

def get_entropy(image):
    image = ensure_uint8(image)
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    probabilities = histogram / histogram.sum()
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))

def get_sobel(image):
    image = ensure_uint8(image).astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f"get_sobel esperaba imagen 2D, recibió {image.shape}")
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    sobel_y = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ], dtype=np.float32)
    gx = convolve2d(image, sobel_x, mode='same', boundary='symm')
    gy = convolve2d(image, sobel_y, mode='same', boundary='symm')
    magnitude = np.sqrt(gx**2 + gy**2)
    return float(np.mean(magnitude))

def evaluate_population(population):
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        fitness_values = list(executor.map(objective_function, population))

    for subject, fit in zip(population, fitness_values):
        subject["fitness"] = fit

    return population


def show_population(population):
    for subject in population:
        x_real = subject['variables'][0]
        y_real = subject['variables'][1]
        aptitud = subject['fitness']
        print(f"Subject: Valor real: ({x_real:.9f}, {y_real:.9f}), Aptitud: {aptitud}")


def get_initial_population():
    population = []
    for _ in range(TAM_POBLACION):
        subject = {}
        subject_x = rng.uniform(LIMITE_INFERIORX, LIMITE_SUPERIORX)
        subject_y = rng.uniform(LIMITE_INFERIORY, LIMITE_SUPERIORY)
        subject['variables'] = np.array([subject_x, subject_y])
        population.append(subject)
    return population


def get_tournament_matrix():
    tournament_matrix = np.zeros((TAM_POBLACION, TOURNAMET_SIZE), dtype=int)
    for i in range(TAM_POBLACION):
        tournament_matrix[i] = rng.choice(TAM_POBLACION, size=TOURNAMET_SIZE, replace=False)
    return tournament_matrix
 
def parent_selection(population, tournament_matrix):
    selected_parents = []
    for i in range(TAM_POBLACION):
        tournament_subjects = [population[j] for j in tournament_matrix[i]]
        best_subject = min(tournament_subjects, key=lambda subj: subj['fitness'])
        selected_parents.append(best_subject)
    return selected_parents


def crossover_sbx(parent1, parent2, prob_cruce):
    child1 = {'variables': np.zeros(2)}
    child2 = {'variables': np.zeros(2)}
    if rng.random() < prob_cruce:
        for j in range(len(parent1['variables'])):
            P1 = parent1['variables'][j]
            P2 = parent2['variables'][j]
            beta = 1 + (2 / P1 - P2) * min(P1 - LI[j], LS[j] - P2) 
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
    

def polinomial_mutation(subject, prob_mutacion):
    if rng.random() < prob_mutacion:
        for j in range(len(subject['variables'])):
            P = subject['variables'][j]
            r = rng.random()
            delta = min(LS[j] - P, P - LI[j]) / (LS[j] - LI[j])
            if r < 0.5:
                delta_q = (2 * r + (1 - 2 * r) * (1 - delta) ** (NM + 1)) ** (1 / (NM + 1)) - 1
            else:
                delta_q = 1 - (2 * (1 - r) + 2 * (r - 0.5) * (1 - delta) ** (NM + 1)) ** (1 / (NM + 1))
            subject['variables'][j] += delta_q * (LS[j] - LI[j])
    return subject


def get_best_global_solution(generations):
    best_subject = None
    for generation in generations:
        for subject in generation:
            if best_subject is None or subject['fitness'] < best_subject['fitness']:
                best_subject = subject
    return best_subject

def save_image(subject, test_run):
    x = subject['variables'][0]
    y = subject['variables'][1]
    transformed_image = get_sigmoid_image(IMAGE_OBJECT, x, y)
    if transformed_image.ndim == 2:
        plt.imshow(transformed_image, cmap='gray')
    else:
        plt.imshow(transformed_image)
    plt.axis('off')
    plt.savefig(f"results/test_run_{test_run}_fecha_{START_DATETIME}.png", bbox_inches='tight', pad_inches=0)
    plt.close()


def add_data_to_csv(run_number, population, prob_mutacion, prob_cruce): 
    best_subject = min(population, key=lambda subj: subj['fitness'])
    worse_subject = max(population, key=lambda subj: subj['fitness'])
    fitness_values = [subj['fitness'] for subj in population]
    median_fitness = np.median(fitness_values)
    std_fitness = np.std(fitness_values)    
    with open(f'results_{START_DATETIME}.csv', 'a') as f:
        if run_number == 0:
            f.write("Fecha, Prueba, Mejor X, Mejor Y, Mejor solucion, Peor X, Peor Y, Peor Solucion, Mediana, Desviacion estandar, Probabilidad de mutacion, Probabilidad de cruce\n")

        f.write(f"{START_DATETIME}, {run_number}, {best_subject['variables'][0]}, {best_subject['variables'][1]}, {best_subject['fitness']}, {worse_subject['variables'][0]}, {worse_subject['variables'][1]}, {worse_subject['fitness']}, {median_fitness}, {std_fitness}, {prob_mutacion}, {prob_cruce}\n")



if __name__ == "__main__":
    generations_data = []
    xyz_data_points = []
    for _ in range(10):
        print(f"Test run: {_}")
        initial_population = get_initial_population()
        generations_data.append({'test_number': _, 'generations': [], 'best_global_solution': None, 'best_global_fitness': None})
        for generation in range(TOT_GENERACIONES):
            print(f"Generation: {generation}")
            best_subject = min(initial_population, key=lambda subj: objective_function(subj))
            
            initial_population = evaluate_population(initial_population)

            tournament_matrix = get_tournament_matrix()
            selected_parents = parent_selection(initial_population, tournament_matrix)
            
            new_population = []
            for i in range(0, TAM_POBLACION, 2):
                parent1 = selected_parents[i]
                parent2 = selected_parents[i + 1]
                child1, child2 = crossover_sbx(parent1, parent2, PROB_CRUCE[_])
                new_population.append(polinomial_mutation(child1, PROB_MUTACION[_]))
                new_population.append(polinomial_mutation(child2, PROB_MUTACION[_]))
            generations_data[-1]['generations'].append(initial_population)
            initial_population = new_population
            #replace a random subject in the new population with the best subject from the previous generation
            random_index = rng.integers(0, TAM_POBLACION)
            initial_population[random_index] = best_subject
        last_generation = generations_data[-1]['generations'][-1]
        best_global_solution = get_best_global_solution(generations_data[-1]['generations'])
        best_global_fitness = objective_function(best_global_solution)
        generations_data[-1]['best_global_solution'] = best_global_solution
        generations_data[-1]['best_global_fitness'] = best_global_fitness
        add_data_to_csv(_, last_generation, PROB_MUTACION[_], PROB_CRUCE[_])



    #search for the best solution among all generations and all test runs
    run_index = None
    best_subject = None
    for test_run in generations_data:
        for generation in test_run['generations']:
            for subject in generation:
                # add each subject's variables and fitness to the xyz_data_points list
                xyz_data_points.append((subject['variables'][0], subject['variables'][1], subject['fitness']))
                if best_subject is None or subject['fitness'] < best_subject['fitness']:
                    best_subject = subject
                    run_index = test_run['test_number']

    best_run = generations_data[run_index]['generations']  
    print(f"Best solution found in test run {run_index}: X = {best_subject['variables'][0]}, Y = {best_subject['variables'][1]}, Fitness = {best_subject['fitness']}")
    save_image(best_subject, run_index)


    #plot and save the xyz_data_points to a csv file
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = [point[0] for point in xyz_data_points]
    ys = [point[1] for point in xyz_data_points]
    zs = [point[2] for point in xyz_data_points]
    ax.scatter(xs, ys, zs, c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')
    plt.savefig(f"plots/xyz_data_points_{START_DATETIME}.png")
    with open(f"plots/xyz_data_points_{START_DATETIME}.csv", 'w') as f:
        f.write("X, Y, Fitness\n")
        for x, y, fitness in xyz_data_points:
            f.write(f"{x}, {y}, {fitness}\n")

