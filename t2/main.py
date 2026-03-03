from random import random
import numpy as np
import pandas as pd
rng = np.random.default_rng()

TAM_POBLACION = 100
TOT_GENERACIONES = 100
PUNTOS_CRUCE = 2
PROB_MUTACION = 0.01 

PRECISION = 3


LIMITE_INFERIORX = -5
LIMITE_SUPERIORX = 5
LIMITE_INFERIORY = -5
LIMITE_SUPERIORY = 5

def get_subject_length(limite_inferior, limite_superior, precision):
    rango = limite_superior - limite_inferior
    return int(np.ceil(np.log2(rango * (10 ** precision))))

X_SUBJECT_LENGTH = get_subject_length(LIMITE_INFERIORX, LIMITE_SUPERIORX, PRECISION)
Y_SUBJECT_LENGTH = get_subject_length(LIMITE_INFERIORY, LIMITE_SUPERIORY, PRECISION)

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
    return x**2 + y**2

population = get_initial_population()
population.append({'variables': np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1])})
for subject in population:
    x_real = bits_to_integer(subject['variables'][:X_SUBJECT_LENGTH])
    y_real = bits_to_integer(subject['variables'][X_SUBJECT_LENGTH:X_SUBJECT_LENGTH+Y_SUBJECT_LENGTH])
    decoded_x = decode_value(x_real, X_SUBJECT_LENGTH, LIMITE_INFERIORX, LIMITE_SUPERIORX)
    decoded_y = decode_value(y_real, Y_SUBJECT_LENGTH, LIMITE_INFERIORY, LIMITE_SUPERIORY)
    aptitud = objective_function(decoded_x, decoded_y)
    print(f"Subject: {subject['variables'][:9]}, Valor real: ({decoded_x:.3f}, {decoded_y:.3f}), Valor entero: ({x_real}, {y_real}), Aptitud: {aptitud}")

def get_tournament_selection(population, tournament_size=2):
    selected = rng.choice(population, size=tournament_size, replace=False)
    best_subject = min(selected, key=lambda subj: subj['aptitud'])
    return best_subject

for i in range(TOT_GENERACIONES):
    print(f"Generación {i+1}")
    for subject in population: