import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.mlab as ml
from scipy.interpolate import griddata

#creating the logistic function
def logistic_func(w, x):
    #x is a feature vector
    #w is the parameter vector
    regression = np.dot(x, w) #finding weighted sum of inputs
    result = 1 / (1 + np.exp(- regression))
    return result


#creating cross entropy function
def cross_entropy_func(w , x, y):
    result = - np.sum (y * np.log(logistic_func(w,x)) + (1-y)* np.log(1- logistic_func(w,x)))
    return result

#gradient function
def gradient_func(w, x, y, learning_rate):
    result = np.dot(x.T, logistic_func(w, x) - y)
    return learning_rate * result


#gradient descent function
def sgd(w,x,y,iterations, learning_rate):
    m = len(y) # size of the training dataset
    cost_history = []
    for _ in range(iterations):
        
        for j in range(m): #loop through every sample
            x_i = x[1,:].reshape((1,len(w)))
            y_i = y[1,:].reshape((1,1))
            w = w - gradient_func(w,x_i,y_i, learning_rate)
            cost = cross_entropy_func(w, x_i, y_i)
        cost_history.append(cost)
        
    return w, cost_history



path = "c:/datasets/acute-nephritis.csv"
data_pd = pd.read_csv(path, header=None)  
print(data_pd)



"""
1) Split dataset from features x[] and class y[]
2) Create/update vectors
    Add 1 at the beggining of every feature vector
    Create w[]
    increase dimmensionality of y[] (from 1D array to column vector)
3) Split the data into training and test 
4) Train the model. 
    Use SGD to create optimized parameters (w[])
        Input in xt, yt, w?, iterations, learning rate
        Loop: However many iterations to train (user define)
            Loop:   Through every sample
                    update w using gradiant funct (derivative) w xi yi learning rate
                    update cost using cross entropy
5)
"""

#seperating the features and labels.
x = data_pd.iloc[:, 0:-1]
y = data_pd.iloc[:, -1]
#conveting to array
x_arr = x.values
y_arr = y.values

# Adding 1 at the begining of every feature vector
x_arr = np.c_[np.ones((x_arr.shape[0], 1)), x_arr]
y_arr = y_arr[:, np.newaxis]
w = np.zeros((x_arr.shape[1],1)) #initializing the parameter vector w.

#creating training test date
X_train, X_test, y_train, y_test = \
                    train_test_split(x_arr, y_arr, test_size=0.20)