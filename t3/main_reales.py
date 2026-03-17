import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np

rng = np.random.default_rng()

TAM_POBLACION = 100
TOT_GENERACIONES = 200
NC = 2
NM = 20
PROB_MUTACION = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
PROB_CRUCE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PRECISION = 5

TOURNAMET_SIZE = 3
LIMITE_INFERIORX = 0
LIMITE_SUPERIORX = 10
LIMITE_INFERIORY = 0
LIMITE_SUPERIORY = 10

RESOLUTION = .1


LI = [1,-1]
LS = [3,5]

def objective_function(x, y):

    #TEST FUNCTION
    #return 0.5 * (x**4 -16 * x**2 + 5*x + y**4 -16 * y**2 + 5*y )

    #######langermann function
    sum = 0
    a = [3, 5, 2, 1, 7]
    b = [5, 2, 1, 4, 9]
    c = [1, 2, 5, 2, 3]
    for i in range(len(a)):
        sum += c[i] * np.cos(np.pi * ((x - a[i]) ** 2 + (y - b[i]) ** 2)) * np.exp(-((x - a[i]) ** 2 + (y - b[i]) ** 2) / np.pi)
    return -sum

    #######DROPWAVE FUNCTION
    # return -(1 + np.cos(12 * np.sqrt(x**2 + y**2))) / (0.5 * (x**2 + y**2) + 2)

    


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
            beta = 1 + (2 / P1 - P2) * min(P1 - LI[j], LS[j] - P2) 
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

def add_data_to_csv(run_number, population, prob_mutacion, prob_cruce): 
    best_subject = min(initial_population, key=lambda subj: objective_function(
            subj['variables'][0], subj['variables'][1]
        ))
    worse_subject = max(initial_population, key=lambda subj: objective_function(
            subj['variables'][0], subj['variables'][1]
        ))
    fitness_values = [objective_function(
        subj['variables'][0], subj['variables'][1]
    ) for subj in population]
    median_fitness = np.median(fitness_values)
    std_fitness = np.std(fitness_values)    
    with open('results.csv', 'a') as f:
        if run_number == 0:
            f.write("Prueba, Mejor X, Mejor Y, Mejor solucion, Peor X, Peor Y, Peor Solucion, Mediana, Desviacion estandar, Probabilidad de mutacion, Probabilidad de cruce\n")

        f.write(f"{run_number}, {best_subject['variables'][0]}, {best_subject['variables'][1]}, {objective_function(best_subject['variables'][0], best_subject['variables'][1])}, {worse_subject['variables'][0]}, {worse_subject['variables'][1]}, {objective_function(worse_subject['variables'][0], worse_subject['variables'][1])}, {median_fitness}, {std_fitness}, {prob_mutacion}, {prob_cruce}\n")

generations_data = []

for _ in range(10):
    initial_population = get_initial_population()

    for generation in range(TOT_GENERACIONES):
        best_subject = min(initial_population, key=lambda subj: objective_function(
            subj['variables'][0], subj['variables'][1]
        ))
        
        tournament_matrix = get_tournament_matrix()
        selected_parents = parent_selection(initial_population, tournament_matrix)
        
        new_population = []
        for i in range(0, TAM_POBLACION, 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            child1, child2 = crossover_sbx(parent1, parent2, PROB_CRUCE[_])
            new_population.append(polinomial_mutation(child1, PROB_MUTACION[_]))
            new_population.append(polinomial_mutation(child2, PROB_MUTACION[_]))
        generations_data.append(initial_population)
        initial_population = new_population
        #replace a random subject in the new population with the best subject from the previous generation
        random_index = rng.integers(0, TAM_POBLACION)
        initial_population[random_index] = best_subject
    add_data_to_csv(_, initial_population, PROB_MUTACION[_], PROB_CRUCE[_])


fig = plt.figure(figsize=(10, 9))


X = np.arange(LIMITE_INFERIORX, LIMITE_SUPERIORX, RESOLUTION)
Y = np.arange(LIMITE_INFERIORY, LIMITE_SUPERIORY, RESOLUTION)
X, Y = np.meshgrid(X, Y)

#calculate Z values for the surface plot using the objective function declared above
Z = objective_function(X, Y)

sub = fig.add_subplot(1, 1, 1)
contour = sub.contourf(X, Y, Z, levels=100, cmap=cm.rainbow, alpha=1)
sub.set_xlim(LIMITE_INFERIORX - 0.01, LIMITE_SUPERIORX + 0.01)
sub.set_ylim(LIMITE_INFERIORY - 0.01, LIMITE_SUPERIORY + 0.01)
#set the title of the plot
sub.set_title("Optimizacion usando AG", fontsize=16)
scatter_2d = sub.scatter([], [], color="red", s=50)


def update(frame):
    population = generations_data[frame]
    #update the title of the plot with the generation number and show the best solution of that generation
    best_x = best_subject['variables'][0]
    best_y = best_subject['variables'][1]
    print(f"Generacion: {frame}, Mejor solucion: ({best_x:.5f}, {best_y:.5f})")
    x = [subj['variables'][0] for subj in population]
    y = [subj['variables'][1] for subj in population]
    scatter_2d.set_offsets(np.c_[x, y])
    return scatter_2d, sub, 

ani = animation.FuncAnimation(fig, update, frames=len(generations_data), interval=500, blit=True)
plt.show()