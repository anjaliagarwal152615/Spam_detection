import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

#naive bayes from scratch 
def gaussian(x, mean, std):
    exponent = np.exp(-(x-mean)**2/(2*std**2))
    return (1 / (np.sqrt(2*np.pi) * std)) * exponent

def naive_bayes(x, mean, std, p):
    return p * gaussian(x, mean, std)

