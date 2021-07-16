from random import random
from copy import deepcopy


def generate_one(dim):
    was_zero = False
    data = []
    for i in range(dim):
        data.append(int(random() > 0.5))
        if (data[i] == 0):
            was_zero = True
    if was_zero:
        return data
    return generate_one(dim)

def add(data_):
    data = deepcopy(data_)
    for i in range(len(data) - 1, -1, -1):
        data[i] += 1
        if (data[i] == 1):
            return data
        data[i] = 0


def generate_data(dim, n):
    array = []
    for i in range(n):
        array.append(generate_one(dim))
    return array

def generate_results(array_):
    array = deepcopy(array_)
    data = []
    for i in range(len(array)):
        data.append(add(array[i]))
    return data

def format(array_):
    array = deepcopy(array_[0])
    for i in range(len(array)):
        if (array[i] < 0.5):
            array[i] = 0
        else:
            array[i] = 1
    print(array)
