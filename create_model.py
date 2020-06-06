import sys
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataset
data_path = './data/data.csv'
dataset = pd.read_csv(data_path, header = None)

X = dataset.iloc[:, 0:62].values
y = dataset.iloc[:, 62].values

# Creating the model
from sklearn.svm import SVC
svm = SVC(C = 100, kernel = 'rbf', gamma = 0.001, random_state = 0)
svm.fit(X, y)

# Saving the Model
filename = 'svm_model.h5'
pickle.dump(svm, open(filename, 'wb'))