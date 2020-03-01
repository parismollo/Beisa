import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from net import train_test_split, train

data = pd.read_csv('winequality-red.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

features = list(data.columns)
features = features[:-1]

X = X/np.amax(X,axis=0)
ymax = np.amax(Y)
y = Y/ymax

st.subheader('Defining network architecture')
neurons = st.selectbox('How many neurons in the hidden layer would you like?', (10, 30, 50, 100))
number_of_iterations = st.slider('Choose the number of iterations', 0, 20000, 10)
learning_rate = st.selectbox('Choose the learning rate', (0.001, 0.005, 0.01))
with st.echo():
    architecture = [
        {"dim_entry": len(features), "dim_output": neurons, "activation": "relu"},
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
