import numpy as np
rng = np.random.default_rng()

TAM_POBLACION = 100
TOT_GENERACIONES = 200
PUNTOS_CRUCE = 2
PROB_MUTACION = 0.01 

PRECISION = 5

TOURNAMET_SIZE = 3
LIMITE_INFERIORX = -5.12
LIMITE_SUPERIORX = 5.12
LIMITE_INFERIORY = -5.12
LIMITE_SUPERIORY = 5.12


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

def objective_function(x, y):
    return 20 + (x ** 2 - 10 * np.cos(2 * np.pi * x)) + (y ** 2 - 10 * np.cos(2 * np.pi * y))


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


def crossover(parent1, parent2):
    crossover_point = rng.integers(1, X_SUBJECT_LENGTH + Y_SUBJECT_LENGTH - 1)
    child1 = np.concatenate((parent1['variables'][:crossover_point], parent2['variables'][crossover_point:]))
    child2 = np.concatenate((parent2['variables'][:crossover_point], parent1['variables'][crossover_point:]))
    return {'variables': child1}, {'variables': child2}

def mutation(subject):
    if rng.random() < PROB_MUTACION:
        mutation_point = rng.integers(0, X_SUBJECT_LENGTH + Y_SUBJECT_LENGTH)
        subject['variables'][mutation_point] = 1 - subject['variables'][mutation_point]
    return subject




initial_population = get_initial_population()

for generation in range(TOT_GENERACIONES):
    print(f"Generation {generation + 1}")
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
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutation(child1))
        new_population.append(mutation(child2))

    initial_population = new_population
    #replace a random subject in the new population with the best subject from the previous generation
    random_index = rng.integers(0, TAM_POBLACION)
    initial_population[random_index] = best_subject

# After all generations, find the best subject in the final population
best_subject = min(initial_population, key=lambda subj: objective_function(
    decode_value(bits_to_integer(subj['variables'][:X_SUBJECT_LENGTH]), X_SUBJECT_LENGTH, LIMITE_INFERIORX, LIMITE_SUPERIORX),
    decode_value(bits_to_integer(subj['variables'][X_SUBJECT_LENGTH:X_SUBJECT_LENGTH+Y_SUBJECT_LENGTH]), Y_SUBJECT_LENGTH, LIMITE_INFERIORY, LIMITE_SUPERIORY)
))
worse_subject = max(initial_population, key=lambda subj: objective_function(
    decode_value(bits_to_integer(subj['variables'][:X_SUBJECT_LENGTH]), X_SUBJECT_LENGTH, LIMITE_INFERIORX, LIMITE_SUPERIORX),
    decode_value(bits_to_integer(subj['variables'][X_SUBJECT_LENGTH:X_SUBJECT_LENGTH+Y_SUBJECT_LENGTH]), Y_SUBJECT_LENGTH, LIMITE_INFERIORY, LIMITE_SUPERIORY)
))

show_population([best_subject, worse_subject])