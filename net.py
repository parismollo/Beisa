import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
import time

def _sigmoid(z):
    return 1/(1+np.exp(-z))

def _relu(z):
    return np.maximum(0,z)


def start_layers(architecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(architecture)
    parameters = {}

    for idx, layer in enumerate(architecture):

        layer_idx = idx + 1

        input_layer_size = layer["dim_entry"]
        output_layer_size = layer["dim_output"]


        parameters['w' + str(layer_idx)] = np.random.randn(
            output_layer_size, input_layer_size)  * 0.1
        parameters['b' + str(layer_idx)] = np.random.randn(
            output_layer_size, 1) * 0.1

    return parameters

def layer_feedforward(previous_activation_value, current_weights, current_bias, activation="relu"):
    current_output = np.dot(current_weights, previous_activation_value) + current_bias

    if activation is "relu":
        activation_function = _relu
    elif activation is "sigmoid":
        activation_function = _sigmoid
    else:
        raise Exception('Function not implemented')

    return activation_function(current_output), current_output


def network_feedforward(X, parameters, architecture):
    memory = {}
    current_activation_value = X

    for idx, layer in enumerate(architecture):
        layer_idx = idx + 1

        previous_activation_value = current_activation_value

        current_activation_function = layer["activation"]

        current_weights = parameters["w" + str(layer_idx)]

        current_bias = parameters["b" + str(layer_idx)]

        current_activation_value, current_output = layer_feedforward(previous_activation_value, current_weights, current_bias, current_activation_function)

        memory["a" + str(idx)] = previous_activation_value
        memory["z" + str(layer_idx)] = current_output

    return current_activation_value, memory

def update(parameters, gradients, architecture, learning_rate):

    for layer_idx, layer in enumerate(architecture, 1):
        parameters["w" + str(layer_idx)] -= learning_rate * gradients["dw" + str(layer_idx)]
        parameters["b" + str(layer_idx)] -= learning_rate * gradients["db" + str(layer_idx)]

    return parameters;

def loss_value(predicted_y, Y):

    m = predicted_y.shape[1]

    loss_value = -1 / m * (np.dot(Y, np.log(predicted_y).T) + np.dot(1 - Y, np.log(1 - predicted_y).T))
    return np.squeeze(loss_value)

def network_backpropagation(predicted_y, Y, memory, parameters, architecture):

    gradients = {}


    Y = Y.reshape(predicted_y.shape)

    delta_previous_activation = - (np.divide(Y, predicted_y) - np.divide(1 - Y, 1 - predicted_y));

    for previous_layer_idx, layer in reversed(list(enumerate(architecture))):

        current_layer_idx = previous_layer_idx + 1

        current_activation_function = layer["activation"]

        delta_current_activation = delta_previous_activation

        previous_activation = memory["a" + str(previous_layer_idx)]
        current_output = memory["z" + str(current_layer_idx)]

        current_weights = parameters["w" + str(current_layer_idx)]
        current_bias = parameters["b" + str(current_layer_idx)]

        delta_previous_activation, delta_current_weights, delta_current_bias = layer_backpropagation(
            delta_current_activation, current_weights, current_bias, current_output, previous_activation, current_activation_function)

        gradients["dw" + str(current_layer_idx)] = delta_current_weights
        gradients["db" + str(current_layer_idx)] = delta_current_bias

    return gradients

def delta_sigmoid(delta_activation, output):
    sig = _sigmoid(output)
    return delta_activation * sig * (1 - sig)

def delta_relu(delta_activation, output):
    delta_output = np.array(delta_activation, copy = True)
    delta_output[output <= 0] = 0;
    return delta_output;

def layer_backpropagation(delta_current_activation, current_weights, current_bias, current_output, previous_activation, current_activation_function="relu"):
    m = previous_activation.shape[1]

    if current_activation_function is "relu":
        current_activation_function = delta_relu
    elif current_activation_function is "sigmoid":
        current_activation_function = delta_sigmoid
    else:
        raise Exception('Function Not Implemented ')

    delta_current_output = current_activation_function(delta_current_activation, current_output)

    delta_current_weights = np.dot(delta_current_output, previous_activation.T) / m
    delta_current_bias = np.sum(delta_current_output, axis=1, keepdims=True) / m
    delta_previous_activation = np.dot(current_weights.T, delta_current_output)

    return delta_previous_activation, delta_current_weights, delta_current_bias

def train(X, Y,X_test,Y_test, architecture, epochs, learning_rate):
    parameters = start_layers(architecture, 2)
    cost_history = []
    cost_history_test = []


    with st.spinner('Wait for it...training network'):

        for i in range(epochs):
            predicted_y, memory = network_feedforward(X, parameters, architecture)

            predicted_y_test, memory2 = network_feedforward(X_test, parameters,
                                                      architecture)

            cost = loss_value(predicted_y, Y)
            cost_history.append(cost)
            test_cost = loss_value(predicted_y_test, Y_test)
            cost_history_test.append(test_cost)


            gradients = network_backpropagation(predicted_y, Y, memory,
                                               parameters, architecture)
            parameters = update(parameters, gradients,
                                          architecture, learning_rate)

            if(i % 50 == 0):
                print(f"Iteration: {i} - loss: {cost} ")
    st.success('Done!')
    return parameters, cost_history, cost_history_test


def main():

    st.sidebar.title('Neural net - bike sharing')
    st.subheader('Loading bike sharing data')
    data_loading = st.text('Loading data...')
    data = pd.read_csv('day.csv')
    data_loading.text('Loading data done!')

    st.write(data)

    data_normalization = st.text('Normalizing data...')
    y = data['cnt'].values
    X = data[['weathersit','temp']].values
    X = X/np.amax(X,axis=0)
    ymax = np.amax(y)
    y = y/ymax

    data_normalization.text('Normalization completed')

    st.subheader('Defining network architecture')
    neurons = st.selectbox('How many neurons in the hidden layer would you like?', (10, 30, 50, 100))
    number_of_iterations = st.slider('Choose the number of iterations', 0, 20000, 10)
    learning_rate = st.selectbox('Choose the learning rate', (0.001, 0.005, 0.01))

    with st.echo():
        architecture = [
            {"dim_entry": 2, "dim_output": neurons, "activation": "relu"},
            {"dim_entry": neurons, "dim_output": 1, "activation": "sigmoid"},
        ]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.43, random_state=42)
    parameters, cost_history, cost_history_test = train(np.transpose(x_train), np.transpose(y_train.reshape((y_train.shape[0], 1))),
                                                                      np.transpose(x_test), np.transpose(y_test.reshape((y_test.shape[0], 1))),
                                                                      architecture, number_of_iterations, learning_rate)


    plt.plot(cost_history)
    plt.plot(cost_history_test, 'r')
    plt.legend(['Training','Test'])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.title('Cost per epochs')
    st.pyplot()

    predicted_y, _ = network_feedforward(np.transpose(x_test), parameters, architecture)
    plt.plot(np.transpose(x_test)[1],ymax*y_test,'.')
    plt.plot(np.transpose(x_test)[1],ymax*predicted_y.reshape([-1,1]),'.r')
    plt.legend(['Expected','Predicted'])
    plt.ylabel('Number of bikes rented')
    plt.xlabel('temperature')
    st.pyplot()


if __name__ == '__main__':
    main()
