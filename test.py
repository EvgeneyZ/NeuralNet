from neural_net import NeuralNet as nn
from data import *

myNN = nn(5, 5, 1, 10)
data = generate_data(5, 1000)
results = generate_results(data)

myNN.back_propogation(data, results)
print(myNN.use([0, 1, 1, 1, 1]))
print(myNN.use([1, 0, 0, 1, 1]))
