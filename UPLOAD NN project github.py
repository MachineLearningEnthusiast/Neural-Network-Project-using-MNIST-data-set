import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Neural_network_structure input is a list of neurons in each layer from left to right. Note structure is made to suit
#Scikit learn dataset which is 8x8 (64 pixels total) so that means the list must take the form of [64,...,10] where
#"..." can take on any number / form of layers.
#Learning factor is the penalty applied to the weights and biases of connections 
#Test fraction is a float between 0 and 1 where 0.5 would mean half of the sample is for testing
#Random state makes it so we get the same data set each time

        
def Convert_each_element_in_y_train_to_an_1x10_output_array(y_train, Data_point):
    y_train_1x10 = np.zeros((1,10))
    index = y_train[Data_point]
    y_train_1x10[0][index] = 1
    return np.transpose(y_train_1x10)


def Convert_each_element_in_y_test_to_an_1x10_output_array(y_test, Data_point):
    y_test_1x10 = np.zeros((1,10))
    index = y_test[Data_point]
    y_test_1x10[0][index] = 1
    return y_test_1x10


def Normalize_data_set(Data_set_to_normalize):
    Normalized_set_point = StandardScaler().fit_transform(Data_set_to_normalize)
    return Normalized_set_point

def Creating_biases_and_weights(Neural_network_structure):
    Weights = {}
    Biases = {}
    for i in range(1, len(Neural_network_structure)):
            Weights[i] = np.random.rand(Neural_network_structure[i], Neural_network_structure[i-1])
            Biases[i] = np.random.rand(Neural_network_structure[i], 1)
            #Biases[i] = random.random()
    return Weights, Biases

def Resetting_update_biases_and_weights(Neural_network_structure):
    Update_w = {}
    Update_b = {}
    for i in range(1, len(Neural_network_structure)):
        Update_w[i] = np.zeros((Neural_network_structure[i], Neural_network_structure[i-1]))
        #Update_b[i] = 0
        Update_b[i] = np.zeros((Neural_network_structure[i], 1))
    return Update_w, Update_b



def Sigma_back_prop_term_out_most_layer(b_last_layer, c_last_layer, y_data_point):
    return -(y_data_point - c_last_layer) * Sigmoid_deriv(b_last_layer)

def Sigma_back_prop_hidden_layers(Weights, b, Sigma_add1):
    return (np.dot(np.transpose(Weights), Sigma_add1)) * Sigmoid_deriv(b)


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def Sigmoid_deriv(x):
    return Sigmoid(x) * (1 - Sigmoid(x))


def Feed_forward(X_single_data_point, Weights, Biases, Neural_network_structure):
    b = {}
    c = {}
    
    for i in range(len(Neural_network_structure)):
        if i == 0:
            c[0] = X_single_data_point             
        else:
            b[i] = np.dot(Weights[i], c[i-1]) + Biases[i]
            c[i] = Sigmoid(b[i])    
    return b, c

def Neural_Network_MNIST(Neural_network_structure, Learning_factor, Test_fraction):
    #Loading digits and splitting them up into datasets
    #Still need to normalize the data.
    digits = load_digits()
    digits_X, digits_y = digits["data"], digits["target"]
    X_train, X_test, y_train, y_test = train_test_split(digits_X, digits_y, test_size = Test_fraction)
    
    X_train = Normalize_data_set(X_train)
    
    Weights, Biases = Creating_biases_and_weights(Neural_network_structure)
    Cost_function_average = []
    Iteration_number = 0
    Training_set_size = len(y_train)
    
    while Iteration_number <= Training_set_size:
        Average_cost = 0
        Update_w, Update_b = Resetting_update_biases_and_weights(Neural_network_structure)
    
        for i in range(Training_set_size):
            Sigma = {}
            b, c = Feed_forward(X_train[i][:, np.newaxis], Weights, Biases, Neural_network_structure)
            
            y_data_point = Convert_each_element_in_y_train_to_an_1x10_output_array(y_train, i)
            #middle 1 was a zero checkkk!!!!
            for j in range(len(Neural_network_structure)-1, -1, -1):
                if j == len(Neural_network_structure) - 1:
                    Average_cost += np.linalg.norm((y_data_point-c[j])) 
                    Sigma[j] = Sigma_back_prop_term_out_most_layer(b[j], c[j], y_data_point) 
                else:
                    if j > 0:
                        Sigma[j] = Sigma_back_prop_hidden_layers(Weights[j+1], b[j], Sigma[j+1]) 
                        if j > -1 and j < 3:
                            Update_b[j+1] += Sigma[j+1]
                            Update_w[j+1] += np.dot(Sigma[j+1], np.transpose(c[j]))
        
        for k in range(len(Neural_network_structure)-1,0,-1):
            Biases[k] -= Learning_factor * ((1 / Training_set_size) * Update_b[k])
            Weights[k] -= Learning_factor * ((1 / Training_set_size) * Update_w[k])
        
        Iteration_number += 1 
        Average_cost = (1 / Training_set_size) * Average_cost
        Cost_function_average.append(Average_cost)
    return Cost_function_average, Weights, Biases, X_test, y_test



Cost_function_average, Weights, Biases, X_test, y_test = Neural_Network_MNIST([64,32,10], 0.25, 0.76)
print(Cost_function_average)
plt.plot(np.arange(0, len(Cost_function_average), 1), Cost_function_average)
plt.xlabel("Iteration Number")
plt.ylabel("Cost")
plt.show()
print(y_test[0:3])
print(X_test[0:3])

def Test_NN(Weights, Biases, X_test, y_test, Neural_network_structure):
    Correctly_predicted = 0
    Training_set_size = len(y_test)
    X_test = Normalize_data_set(X_test)
    
    for i in range(len(y_test)):
        b, c = Feed_forward(X_test[i][:, np.newaxis], Weights, Biases, Neural_network_structure)
        NN_predicted_index = np.argmax(c[len(Neural_network_structure)-1])
        if NN_predicted_index == y_test[i]:
            Correctly_predicted += 1
    Ratio_of_correctly_predicted_to_total_tests = Correctly_predicted / Training_set_size
    return Correctly_predicted, Ratio_of_correctly_predicted_to_total_tests
        
Correctly_predicted, Ratio_of_correctly_predicted_to_total_tests = Test_NN(Weights, Biases, X_test, y_test, [64,32,10])      
print(Correctly_predicted, Ratio_of_correctly_predicted_to_total_tests)
