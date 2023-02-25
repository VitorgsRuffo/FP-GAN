print()
print(format('How to predict a timeseries using GRU in Keras','*^92'))

# load libraries
import pandas, time
import numpy as np
from keras.layers import GRU
from keras.layers.core import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# load the dataset
dataset = np.array([[1, 2], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17]])
train_dataset = dataset.astype('float32')
print(train_dataset)
print("\n\n")


# Window -> X timestep back
step_back = 2
X_train, Y_train = [], []
for i in range(len(train_dataset)-step_back - 1):
    a = train_dataset[i:(i+step_back), :]
    X_train.append(a)
    Y_train.append(train_dataset[i + step_back, :])
X_train = np.array(X_train); Y_train = np.array(Y_train);
print(X_train); print(); print(Y_train);            
print("\n\n")

# -----------------------------------------------------------
# reshape input to be [samples (windows), time steps (window size), features (window element size)]
# -----------------------------------------------------------    
X_train = np.reshape(X_train, (X_train.shape[0], 2, 2))

print(X_train); print(); print(Y_train);            

print("\n\n")
X_train = np.reshape(X_train, (X_train.shape[0], -1))
print(X_train)
print(Y_train)
