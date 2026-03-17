import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np

from main2 import X_SUBJECT_LENGTH
rng = np.random.default_rng()

TAM_POBLACION = 100
TOT_GENERACIONES = 200
NC = 2
PROB_MUTACION = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
PROB_CRUCE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PRECISION = 5

TOURNAMET_SIZE = 3
LIMITE_INFERIORX = -5.12
LIMITE_SUPERIORX = 5.12
LIMITE_INFERIORY = -5.12
LIMITE_SUPERIORY = 5.12


LI = [1,-1]
LS = [3,5]

def objective_function(x, y):
    return 0.5 * (x**4 -16 * x**2 + 5*x + y**4 -16 * y**2 + 5*y )


def show_population(population):
    for subject in population:
        x_real = subject['variables'][0]
        y_real = subject['variables'][1]
        aptitud = objective_function(x_real, y_real)
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
        best_subject = min(tournament_subjects, key=lambda subj: objective_function(
            subj['variables'][0], subj['variables'][1]
        ))
        selected_parents.append(best_subject)
    return selected_parents


def crossover_sbx(parent1, parent2, prob_cruce):
    child1 = {'variables': np.zeros(2)}
    child2 = {'variables': np.zeros(2)}
    if rng.random() < prob_cruce:
        for j in range(len(parent1['variables'])):
            P1 = parent1['variables'][j]
            P2 = parent2['variables'][j]
            beta = 1 + (2 * min(P1 - LI[j], LS[j] - P2) / (P1 - P2))
            alpha = 2 - abs(beta) ** -(NC + 1)
            if rng.random() < 1 / alpha:
                beta_q = (rng.random() * alpha) ** (1 / (NC + 1))
            else:
                beta_q = (1 / (2 - rng.random() * alpha)) ** (1 / (NC + 1))
            child1_j = 0.5 * ((P1 + P2) - beta_q * abs(P1 - P2))
            child2_j = 0.5 * ((P1 + P2) + beta_q * abs(P1 - P2))
            child1['variables'][j] = child1_j
            child2['variables'][j] = child2_j
    else:
        child1['variables'] = parent1['variables'].copy()
        child2['variables'] = parent2['variables'].copy()
    return child1, child2
    

def mutation(subject, prob_mutacion):
    if rng.random() < prob_mutacion:
        mutation_point = rng.integers(0, X_SUBJECT_LENGTH + Y_SUBJECT_LENGTH)
        subject['variables'][mutation_point] = 1 - subject['variables'][mutation_point]
    return subject


def add_data_to_csv(run_number, population, prob_mutacion, prob_cruce): 
    best_subject = min(population, key=lambda subj: objective_function(
        decode_value(bits_to_integer(subj['variables'][:X_SUBJECT_LENGTH]), X_SUBJECT_LENGTH, LIMITE_INFERIORX, LIMITE_SUPERIORX),
        decode_value(bits_to_integer(subj['variables'][X_SUBJECT_LENGTH:X_SUBJECT_LENGTH+Y_SUBJECT_LENGTH]), Y_SUBJECT_LENGTH, LIMITE_INFERIORY, LIMITE_SUPERIORY)
    ))
    worse_subject = max(population, key=lambda subj: objective_function(
        decode_value(bits_to_integer(subj['variables'][:X_SUBJECT_LENGTH]), X_SUBJECT_LENGTH, LIMITE_INFERIORX, LIMITE_SUPERIORX),
        decode_value(bits_to_integer(subj['variables'][X_SUBJECT_LENGTH:X_SUBJECT_LENGTH+Y_SUBJECT_LENGTH]), Y_SUBJECT_LENGTH, LIMITE_INFERIORY, LIMITE_SUPERIORY)
    ))
    fitness_values = [objective_function(
        decode_value(bits_to_integer(subj['variables'][:X_SUBJECT_LENGTH]), X_SUBJECT_LENGTH, LIMITE_INFERIORX, LIMITE_SUPERIORX),
        decode_value(bits_to_integer(subj['variables'][X_SUBJECT_LENGTH:X_SUBJECT_LENGTH+Y_SUBJECT_LENGTH]), Y_SUBJECT_LENGTH, LIMITE_INFERIORY, LIMITE_SUPERIORY)
    ) for subj in population]
    median_fitness = np.median(fitness_values)
    std_fitness = np.std(fitness_values)    
    with open('results.csv', 'a') as f:
        if run_number == 0:
            f.write("Prueba, Mejor X, Mejor Y, Mejor solucion, Peor X, Peor Y, Peor Solucion, Mediana, Desviacion estandar, Probabilidad de mutacion, Probabilidad de cruce\n")

        f.write(f"{run_number}, {decode_value(bits_to_integer(best_subject['variables'][:X_SUBJECT_LENGTH]), X_SUBJECT_LENGTH, LIMITE_INFERIORX, LIMITE_SUPERIORX)}, {decode_value(bits_to_integer(best_subject['variables'][X_SUBJECT_LENGTH:X_SUBJECT_LENGTH+Y_SUBJECT_LENGTH]), Y_SUBJECT_LENGTH, LIMITE_INFERIORY, LIMITE_SUPERIORY)}, {objective_function(decode_value(bits_to_integer(best_subject['variables'][:X_SUBJECT_LENGTH]), X_SUBJECT_LENGTH, LIMITE_INFERIORX, LIMITE_SUPERIORX), decode_value(bits_to_integer(best_subject['variables'][X_SUBJECT_LENGTH:X_SUBJECT_LENGTH+Y_SUBJECT_LENGTH]), Y_SUBJECT_LENGTH, LIMITE_INFERIORY, LIMITE_SUPERIORY))}, {decode_value(bits_to_integer(worse_subject['variables'][:X_SUBJECT_LENGTH]), X_SUBJECT_LENGTH, LIMITE_INFERIORX, LIMITE_SUPERIORX)}, {decode_value(bits_to_integer(worse_subject['variables'][X_SUBJECT_LENGTH:X_SUBJECT_LENGTH+Y_SUBJECT_LENGTH]), Y_SUBJECT_LENGTH, LIMITE_INFERIORY, LIMITE_SUPERIORY)}, {objective_function(decode_value(bits_to_integer(worse_subject['variables'][:X_SUBJECT_LENGTH]), X_SUBJECT_LENGTH, LIMITE_INFERIORX, LIMITE_SUPERIORX), decode_value(bits_to_integer(worse_subject['variables'][X_SUBJECT_LENGTH:X_SUBJECT_LENGTH+Y_SUBJECT_LENGTH]), Y_SUBJECT_LENGTH, LIMITE_INFERIORY, LIMITE_SUPERIORY))}, {median_fitness}, {std_fitness}, {prob_mutacion}, {prob_cruce}\n")

generations_data = []

for _ in range(10):
    initial_population = get_initial_population()

    for generation in range(TOT_GENERACIONES):
        best_subject = min(initial_population, key=lambda subj: objective_function(
            decode_value(bits_to_integer(subj['variables'][:X_SUBJECT_LENGTH]), X_SUBJECT_LENGTH, LIMITE_INFERIORX, LIMITE_SUPERIORX),
            decode_value(bits_to_integer(subj['variables'][X_SUBJECT_LENGTH:X_SUBJECT_LENGTH+Y_SUBJECT_LENGTH]), Y_SUBJECT_LENGTH, LIMITE_INFERIORY, LIMITE_SUPERIORY)
        ))
        
        tournament_matrix = get_tournament_matrix()
        selected_parents = parent_selection(initial_population, tournament_matrix)
        
        new_population = []
        for i in range(0, TAM_POBLACION, 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            child1, child2 = crossover(parent1, parent2, PROB_CRUCE[_])
            new_population.append(mutation(child1, PROB_MUTACION[_]))
            new_population.append(mutation(child2, PROB_MUTACION[_]))
        generations_data.append(initial_population)
        initial_population = new_population
        #replace a random subject in the new population with the best subject from the previous generation
        random_index = rng.integers(0, TAM_POBLACION)
        initial_population[random_index] = best_subject
    add_data_to_csv(_, initial_population, PROB_MUTACION[_], PROB_CRUCE[_])


fig = plt.figure(figsize=(10, 9))


X = np.arange(LIMITE_INFERIORX, LIMITE_SUPERIORX, .12)
Y = np.arange(LIMITE_INFERIORY, LIMITE_SUPERIORY, .12)
X, Y = np.meshgrid(X, Y)

#calculate Z values for the surface plot using the objective function declared above
Z = objective_function(X, Y)

sub = fig.add_subplot(1, 1, 1)
contour = sub.contourf(X, Y, Z, levels=50, cmap=cm.turbo, alpha=0.9)
sub.set_xlim(LIMITE_INFERIORX - 0.01, LIMITE_SUPERIORX + 0.01)
sub.set_ylim(LIMITE_INFERIORY - 0.01, LIMITE_SUPERIORY + 0.01)
#set the title of the plot
sub.set_title("Optimizacion usando AG", fontsize=16)
scatter_2d = sub.scatter([], [], color="red", s=50)


def update(frame):
    population = generations_data[frame]
    #update the title of the plot with the generation number and show the best solution of that generation
    best_subject = min(population, key=lambda subj: objective_function(
        decode_value(bits_to_integer(subj['variables'][:X_SUBJECT_LENGTH]), X_SUBJECT_LENGTH, LIMITE_INFERIORX, LIMITE_SUPERIORX),
        decode_value(bits_to_integer(subj['variables'][X_SUBJECT_LENGTH:X_SUBJECT_LENGTH+Y_SUBJECT_LENGTH]), Y_SUBJECT_LENGTH, LIMITE_INFERIORY, LIMITE_SUPERIORY)
    ))
    best_x = decode_value(bits_to_integer(best_subject['variables'][:X_SUBJECT_LENGTH]), X_SUBJECT_LENGTH, LIMITE_INFERIORX, LIMITE_SUPERIORX)
    best_y = decode_value(bits_to_integer(best_subject['variables'][X_SUBJECT_LENGTH:X_SUBJECT_LENGTH+Y_SUBJECT_LENGTH]), Y_SUBJECT_LENGTH, LIMITE_INFERIORY, LIMITE_SUPERIORY)
    print(f"Generacion: {frame}, Mejor solucion: ({best_x:.5f}, {best_y:.5f})")
    x = [decode_value(bits_to_integer(subj['variables'][:X_SUBJECT_LENGTH]), X_SUBJECT_LENGTH, LIMITE_INFERIORX, LIMITE_SUPERIORX) for subj in population]
    y = [decode_value(bits_to_integer(subj['variables'][X_SUBJECT_LENGTH:X_SUBJECT_LENGTH+Y_SUBJECT_LENGTH]), Y_SUBJECT_LENGTH, LIMITE_INFERIORY, LIMITE_SUPERIORY) for subj in population]
    scatter_2d.set_offsets(np.c_[x, y])
    return scatter_2d, sub, 

ani = animation.FuncAnimation(fig, update, frames=len(generations_data), interval=500, blit=True)
plt.show()