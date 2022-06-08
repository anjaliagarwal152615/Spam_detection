import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#logistic regression from scratch 
def logistic_regression(X, y, learning_rate, num_iterations):
    #initialize theta
    theta = np.zeros(X.shape[1])
    #iterate through the number of iterations
    for i in range(num_iterations):
        #prediction
        y_pred = sigmoid(np.dot(X, theta))
        #update theta
        theta = theta - learning_rate * np.dot(X.T, (y_pred - y))
    return theta
