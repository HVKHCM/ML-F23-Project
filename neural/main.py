import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
import mnist_loader
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt

training_data, validation_data, test_data = mnist_loader.load_data_wrapper() 
training_data = list(training_data) 
test_data = list(test_data) 

X_train = []
y_train = []

for i in training_data:
    X_train.append(i[0])
    y_train.append(i[1])

X_test = []
y_test = []

for i in test_data:
    X_test.append(i[0])
    y_test.append(i[1])

unit1 = np.arange(10, 101, 20)
unit2 = np.arange(10, 61, 20)
lr = [0.01,0.1,0.5,0.9]
epoch = 10

loss_fn = nn.CrossEntropyLoss()

train_error, accuracy_list,configuration_list = utils.unit_optimize(epochs=epoch, loss_fun=loss_fn,X_train=X_train, y_train=y_train, 
                                   X_test=X_test, y_test=y_test, unit1_range=unit1, unit2_range=unit2, lr_range=lr)