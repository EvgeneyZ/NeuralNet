import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
import sys
import pygame

#np.random.seed(1);

class NeuralNet:

    def __init__(self, input_size, output_size, number_of_hidden_layers, *layers_size, alpha=0.2, print_errors=True, 
            include_bias=False, warning_counter=3):
    
        #debug params    
        self.print_errors = print_errors
        self.error_log = []

        #sizes
        self.include_bias = include_bias
        self.input_size = input_size + int(self.include_bias)
        self.output_size = output_size

        self.number_of_hidden_layers = number_of_hidden_layers

        if (len(layers_size) != number_of_hidden_layers):
            self.throw_error("WARNING! Number of layers and length of size array are not equal! \
                Set number_of_layers = len(layers_size)")
            self.number_of_hidden_layers = len(layers_size)

        self.layers_size = [self.input_size]
        for i in range(self.number_of_hidden_layers):
            self.layers_size.append(layers_size[i])
        self.layers_size.append(self.output_size)

        self.number_of_layers = self.number_of_hidden_layers + 2

        #weights
        self.weights = []
        self.init_all_weights()

        #other params
        self.alpha = alpha
        self.warning_counter_max = warning_counter
        self.warning_counter = self.warning_counter_max
        
        #visuals
        self.visuals = False
        self.screen = -1
        self.visualize()

        #data   
        self.input_data = []
        self.layers_data = [0] * self.number_of_layers
        self.output_data = []

        self.layers_errors = [0] * (self.number_of_layers - 1)

        #constants
        self.ERROR = 1
        self.SUCCESS = 0

    
    def init_all_weights(self):
        self.weights = []
        for i in range(1, self.number_of_layers):
            self.weights.append(self.generate_weights(self.layers_size[i - 1], self.layers_size[i]))

            
    def generate_weights(self, input_size, output_size):
        return 2 * np.random.random((input_size, output_size)) - 1


    def relu(self, x):
        return (x > 0) * x


    def relu2deriv(self, x):
        return x > 0

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    
    def use(self, data_):
        data = deepcopy(data_)
        if (self.include_bias):
            data.append(1)
        if (len(data) != self.input_size):
            self.throw_error("ERROR! Wrong number of inputs!")
            return self.ERROR

        self.input_data = np.array([data])
        self.output_data = self.parse_through_all_layers(self.input_data)
        
        return self.output_data


    def check_back_propogation_inputs(self, data, results):
        if (len(results) == 0):
            self.throw_error("ERROR! Empty inputs list!")
            return self.ERROR

        if (len(results) == 0):
            self.throw_error("ERROR! Empty results list!")
            return self.ERROR

        if (len(results) != len(data)):
            self.throw_error("WARNING! Inputs list size is not equal to results list size!\n" +
                    "Using minimal size")
            if (len(results) > len(data)):
                results = results[:len(data)]
            else:
                data = data[:len(results)]

        for i in range(len(data)):
            if (len(results[i]) != len(self.weights[-1][0])):
                self.throw_error("ERROR! Wrong size of output data!")
                return self.ERROR
 
            if (type(self.use(data[i])) == int):
                return self.ERROR


        return self.SUCCESS
    

    def back_propogation(self, data, results, iterations=100, error_goal=0.01, test_data_percents=0.2):
       
        if (self.check_back_propogation_inputs(data, results) == self.ERROR):
            return self.ERROR

        combine = []
        for i in range(len(data)):
            combine.append([data[i], results[i]])

        combine = np.array(combine)

        train_set, test_set = train_test_split(combine, test_size=0.2, random_state=42)
        
        train_set_data = train_set[:, 0]
        train_set_results = train_set[:, 1]

        test_set_data = test_set[:, 0]
        test_set_results = test_set[:, 1]

        iteration = 0
        last_error = -1
        while True:            
            self.back_propogation_iteration(train_set_data, train_set_results)
            error = self.get_error(test_set_data, test_set_results)
            
            if (last_error == -1 or last_error > error):
                last_error = error
            else:
                self.warning_counter -= 1
                last_error = error
                if (self.warning_counter == 0):
                    print("Warning!")
                    self.warning_counter = self.warning_counter_max
                    break

            iteration += 1
            if (iteration % 10 == 9):
                print("Iteration:", iteration, "Error:", error)
            if (error_goal != 0):
                if (error < error_goal):
                    break
            else:
                if (iteration >= iterations):
                    break

            self.visual_me()

        return self.use(data[0])


    def back_propogation_iteration(self, data, results):
        error = 0

        layers_delta = [0] * (self.number_of_layers - 1)
        
        for i in range(len(data)):
            outputs = self.use(data[i])
            error += np.sum((outputs - results[i]) ** 2)
            layers_delta[-1] = (outputs - results[i])

            for j in range(self.number_of_layers - 3, -1, -1):
                layers_delta[j] = layers_delta[j + 1].dot(self.weights[j + 1].T) * self.sigmoid_deriv(self.layers_data[j + 1])

            for j in range(self.number_of_layers - 2, -1, -1):
                self.weights[j] -= self.alpha * self.layers_data[j].T.dot(layers_delta[j])
        
        return error         

    def get_error(self, data, results):
        error = 0
        for i in range(len(data)):
            outputs = self.use(data[i])
            error += np.sum((outputs - results[i]) ** 2)

        return error


    def parse_through_all_layers(self, inputs):
        self.layers_data[0] = deepcopy(inputs)
        for i in range(1, self.number_of_layers):
            self.layers_data[i] = self.parse_through_layer(self.layers_data[i - 1], i - 1)

        return self.layers_data[-1]          


    def parse_through_layer(self, inputs, layer_num):
        if (layer_num >= self.number_of_layers - 1):
            self.throw_error("ERROR! Layer number out of bounds!")
            return

        weights = self.weights[layer_num]
        if (layer_num == self.number_of_layers - 2):
            return np.dot(inputs, weights)
        return self.sigmoid(np.dot(inputs, weights))


    def throw_error(self, error_text):
        self.error_log.append(error_text)
        if (self.print_errors):
            print(error_text)


    def print_error_log(self, clear=False):
        if (len(self.error_log) == 0):
            print("No errors")
            return

        for i in range(len(self.error_log)):
            print(self.error_log[i])

        if (clear):
            self.error_log = []


    def visualize(self, status=True):
        self.visuals = status

        if (self.visuals):
            pygame.init()
            self.screen = pygame.display.set_mode((1000, 600))
            self.width = 1000
            self.height = 600

    def visual_me(self):
        if (not self.visuals):
            return
        
        for event in pygame.event.get():
            if event == pygame.QUIT:
                sys.exit()
    
        self.screen.fill((255, 255, 255))
        
        cur_x = 0

        for num in range(self.number_of_layers):
            radius = self.height / (2 * self.layers_size[num])
            cur_y = 0
            for i in range(self.layers_size[num]):
                pygame.draw.circle(self.screen, (0, 0, 0), (cur_x + radius, cur_y + radius), radius)
                
                if (num != self.number_of_layers - 1):
                    next_radius = self.height / (2 * self.layers_size[num + 1])
                    for j in range(len(self.weights[num][i])):
                        if (self.weights[num][i][j] > 0):
                            color = (255, 0, 0)
                        else:
                            color = (0, 255, 0)
                        pygame.draw.line(self.screen, color, (cur_x + radius, cur_y + radius), (cur_x + 2 * radius + 60, 2 * next_radius * j + next_radius))
                cur_y += 2 * radius
            cur_x += 2 * radius + 60
        
        pygame.display.flip()

    def help(self):
        print("In development...")
