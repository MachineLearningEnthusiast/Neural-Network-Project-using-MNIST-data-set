import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train, X_test = np.reshape(X_train, (60000,784)), np.reshape(X_test, (10000, 784))

Y_train, Y_test = np.reshape(Y_train, (60000, 1)), np.reshape(Y_test, (10000, 1))

X_train_average = np.average(X_train, axis = 1)
X_train_average = np.expand_dims(X_train_average, axis = 1)

X_test_average = np.average(X_test, axis = 1)
X_test_average = np.expand_dims(X_test_average, axis = 1)

X_train = np.subtract(X_train, X_train_average)
X_test = np.subtract(X_test, X_test_average)

X_test_sd = np.std(X_test, axis = 1)
X_test_sd = np.expand_dims(X_test_sd, axis = 1)
X_test = np.true_divide(X_test, X_test_sd)

X_train_sd = np.std(X_train, axis = 1)
X_train_sd = np.expand_dims(X_train_sd, axis = 1)
X_train = np.true_divide(X_train, X_train_sd)

 
X_test = np.expand_dims(X_test, axis = 2)
X_train = np.expand_dims(X_train, axis = 2)

Zip_test = list(zip(X_test, Y_test))
Zip_train = list(zip(X_train, Y_train))

def Sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def Sigmoid_deriv(z):
    return Sigmoid(z)*(1-Sigmoid(z))

class Neural_network:
    def __init__(self, NN_structure):
        self.Biases = [np.random.rand(a, 1) for a in NN_structure[1:]]
        self.Weights = [np.random.rand(b, a) for a, b in zip(NN_structure[:-1],NN_structure[1:])]
        self.NN_number_of_layers = len(NN_structure)
        self.NN_structure = NN_structure
        
    def Making_batches(self, Batch_size):
        random.shuffle(Zip_train)
        Batches_train = [Zip_train[i:i+Batch_size] for i in range(0, len(Zip_train), Batch_size)]
        return Batches_train
        
    #Where c is the input layer
    def Feedforward(self, c):
        for w, b in zip(self.Weights, self.Biases):            
            c = Sigmoid(np.dot(w, c) + b)
        return c
    
    def Matrix_Y(self, Y):
        Y_matrix = np.zeros((10,1))
        Y_matrix[Y] = 1
        return Y_matrix
           
    def Cost_derivative(self, Y, Last_layer_activation):
        return (Last_layer_activation - self.Matrix_Y(Y))
           
    def Back_Propagation(self, X_data_point, Y_data_point):
        Vectors = []
        Activation = X_data_point
        List_of_activations = [X_data_point]
        Change_weights = [np.zeros(np.shape(p)) for p in self.Weights] 
        Change_biases = [np.zeros(np.shape(q)) for q in self.Biases]
        for w, b in zip(self.Weights, self.Biases):
            Vector = np.dot(w, Activation) + b
            Vectors.append(Vector)
            Activation = Sigmoid(Vector)
            List_of_activations.append(Activation)
        Delta = self.Cost_derivative(Y_data_point, List_of_activations[-1]) * Sigmoid_deriv(Vectors[-1])
        Change_weights[-1] = np.dot(Delta, np.transpose(List_of_activations[-2]))      
        Change_biases[-1] = Delta
        for i in range(self.NN_number_of_layers-2, 0, -1):
            Vector = Vectors[i-1]
            Activation = Sigmoid_deriv(Vector)
            Delta = np.dot(np.transpose(self.Weights[i]), Delta) * Activation
            Change_weights[i-1] = np.dot(Delta, np.transpose(List_of_activations[i-1]))
            Change_biases[i-1] = Delta
        return Change_weights, Change_biases
  
    def SGD(self, Batch_size, Learning_rate):
        Batches = self.Making_batches(Batch_size)
        Batch_number = 0
        for i in Batches:
            Batch_number += 1
            Correctly_predicted = 0 
            Sum_change_weights = [np.zeros(np.shape(n)) for n in self.Weights]
            Sum_change_biases = [np.zeros(np.shape(o)) for o in self.Biases]  
            for j in range(len(i)):
                Change_weights, Change_biases = self.Back_Propagation(i[j][0], i[j][1])
                Sum_change_weights = [u + h for u,h in zip(Change_weights, Sum_change_weights)]
                Sum_change_biases = [u + h for u,h in zip(Change_biases, Sum_change_biases)]
            self.Weights = [c - ((Learning_rate / Batch_size) * v) for c,v in zip(self.Weights, Sum_change_weights)]
            self.Biases = [c - ((Learning_rate / Batch_size) * v) for c,v in zip(self.Biases, Sum_change_biases)]
            for k, l in Zip_test:                
                c = self.Feedforward(k)
                if l[0] == np.argmax(c):
                    Correctly_predicted += 1                
            print("Batch {0} correctly predicted {1} / 10000".format(Batch_number, Correctly_predicted))
                       
NN = Neural_network([784,30,10])
NN.SGD(10,10)
