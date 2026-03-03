from random import random
import numpy as np
import pandas as pd
rng = np.random.default_rng()

TAM_POBLACION = 100
TOT_GENERACIONES = 100
PUNTOS_CRUCE = 2
PROB_MUTACION = 0.01 

X_SUBJECT_LENGTH = 14
Y_SUBJECT_LENGTH = 14

LIMITE_INFERIOR = -5
LIMITE_SUPERIOR = 5



def get_initial_population():
    population = []
    for _ in range(TAM_POBLACION):
        subject_x = rng.integers(0, 2, size=X_SUBJECT_LENGTH)
        subject_y = rng.integers(0, 2, size=Y_SUBJECT_LENGTH)
        population.append(np.concatenate((subject_x, subject_y)))
    return population

def bits_to_integer(bits):
    #gets a list of bits and converts it to an integer
    value = 0
    for index, bit in enumerate(bits):
        value += bit * (2 ** (len(bits) - index - 1))
    return value

def decode_value(integer_value, variable_length):
    return LIMITE_INFERIOR + ((integer_value / (2 ** variable_length - 1)) * (LIMITE_SUPERIOR - LIMITE_INFERIOR))

population = get_initial_population()
population.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
for subject in population:
    x_real = bits_to_integer(subject[:X_SUBJECT_LENGTH])
    y_real = bits_to_integer(subject[X_SUBJECT_LENGTH:X_SUBJECT_LENGTH+Y_SUBJECT_LENGTH])
    decoded_x = decode_value(x_real, X_SUBJECT_LENGTH)
    decoded_y = decode_value(y_real, Y_SUBJECT_LENGTH)
    print(f"Subject: {subject}, Valor real: ({decoded_x}, {decoded_y}), Valor entero: ({x_real}, {y_real})")
