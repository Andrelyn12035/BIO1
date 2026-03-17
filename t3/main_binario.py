import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import numpy as np
rng = np.random.default_rng()

TAM_POBLACION = 100
TOT_GENERACIONES = 200
PUNTOS_CRUCE = 2
PROB_MUTACION = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
PROB_CRUCE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PRECISION = 5

TOURNAMET_SIZE = 3
LIMITE_INFERIORX = -5.12
LIMITE_SUPERIORX = 5.12
LIMITE_INFERIORY = -5.12
LIMITE_SUPERIORY = 5.12


def objective_function(x, y):
    return 0.5 * (x**4 -16 * x**2 + 5*x + y**4 -16 * y**2 + 5*y )



def get_subject_length(limite_inferior, limite_superior, precision):
    rango = limite_superior - limite_inferior
    return int(np.ceil(np.log2(rango * (10 ** precision))))

X_SUBJECT_LENGTH = get_subject_length(LIMITE_INFERIORX, LIMITE_SUPERIORX, PRECISION)
Y_SUBJECT_LENGTH = get_subject_length(LIMITE_INFERIORY, LIMITE_SUPERIORY, PRECISION)

def show_population(population):
    for subject in population:
        x_real = bits_to_integer(subject['variables'][:X_SUBJECT_LENGTH])
        y_real = bits_to_integer(subject['variables'][X_SUBJECT_LENGTH:X_SUBJECT_LENGTH+Y_SUBJECT_LENGTH])
        decoded_x = decode_value(x_real, X_SUBJECT_LENGTH, LIMITE_INFERIORX, LIMITE_SUPERIORX)
        decoded_y = decode_value(y_real, Y_SUBJECT_LENGTH, LIMITE_INFERIORY, LIMITE_SUPERIORY)
        aptitud = objective_function(decoded_x, decoded_y)
        print(f"Subject: {subject['variables'][:]}\n, Valor decodificado: ({decoded_x:.9f}, {decoded_y:.9f}), Aptitud: {aptitud}")


def get_initial_population():
    population = []
    for _ in range(TAM_POBLACION):
        subject = {}
        subject_x = rng.integers(0, 2, size=X_SUBJECT_LENGTH)
        subject_y = rng.integers(0, 2, size=Y_SUBJECT_LENGTH)
        subject['variables'] = np.concatenate((subject_x, subject_y))
        population.append(subject)
    return population

def bits_to_integer(bits):
    #gets a list of bits and converts it to an integer
    value = 0
    for index, bit in enumerate(bits):
        value += bit * (2 ** (len(bits) - index - 1))
    return value

def decode_value(integer_value, variable_length, limite_inferior, limite_superior):
    return limite_inferior + ((integer_value / (2 ** variable_length - 1)) * (limite_superior - limite_inferior))



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
            decode_value(bits_to_integer(subj['variables'][:X_SUBJECT_LENGTH]), X_SUBJECT_LENGTH, LIMITE_INFERIORX, LIMITE_SUPERIORX),
            decode_value(bits_to_integer(subj['variables'][X_SUBJECT_LENGTH:X_SUBJECT_LENGTH+Y_SUBJECT_LENGTH]), Y_SUBJECT_LENGTH, LIMITE_INFERIORY, LIMITE_SUPERIORY)
        ))
        selected_parents.append(best_subject)
    return selected_parents


def crossover(parent1, parent2, prob_cruce):
    if rng.random() < prob_cruce:
        crossover_point = rng.integers(1, X_SUBJECT_LENGTH + Y_SUBJECT_LENGTH - 1)
        child1 = np.concatenate((parent1['variables'][:crossover_point], parent2['variables'][crossover_point:]))
        child2 = np.concatenate((parent2['variables'][:crossover_point], parent1['variables'][crossover_point:]))
    else:
        child1 = parent1['variables'].copy()
        child2 = parent2['variables'].copy()
    return {'variables': child1}, {'variables': child2}

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