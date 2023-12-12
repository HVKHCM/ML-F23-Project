import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mlxtend.evaluate import bias_variance_decomp
import pandas as pd
import numpy as np
import torchviz

# Neural network from scratch using pytorch nn module
class Net(nn.Module):
    def __init__(self, unit1, unit2):
        super().__init__()
        self.layers = nn.ModuleDict()

        self.layers["input"] = nn.Linear(in_features=784, out_features=unit1)
        
        self.layers["hidden_0"] = nn.Linear(in_features=unit1, out_features=unit2)

        self.layers["hidden_1"] = nn.Linear(in_features=unit2, out_features=10)
    

    def forward(self,x):
        x = self.layers["input"](x)

        for i in range(2):
            x = F.sigmoid(self.layers[f"hidden_{i}"](x))

        out = F.softmax(x)
        
        return out

# Function to find and write out data for each parameters setting
def unit_optimize(epochs, loss_fun, X_train, y_train, X_test, y_test, unit1_range, unit2_range, lr_range):
    f = open("data.csv", "a")
    f.write("epochs,firstHU,secondHU,lr,acc,trainAcc\n")
    f.close()
    accuracy_list = []
    all_config = []
    train_error = []
    count = 0
    for i in unit1_range:
        for j in unit2_range:
                for l in lr_range:
                    
                    print("Config: {}".format(count + 1))
                    config = []
                    config.append(i)
                    config.append(j)
                    config.append(l)
                    model = Net(i, j)
                    optimizer = optim.SGD(model.parameters(), lr=l)
                    trained_model = train_model(model,epochs, loss_fun, optimizer, X_train, y_train)
                    accuracy_list.append(eval_model(trained_model, X_test, y_test))
                    train_error.append(eval_model_train(trained_model, X_train, y_train))
                    f = open("data.csv", "a")
                    f.write("{},{},{},{},{},{}\n".format(epochs,i,j,l,accuracy_list[-1],train_error[-1]))
                    f.close()
                    all_config.append(config)
                    count += 1
    return train_error, accuracy_list, all_config

# Training loop for neural network
def train_model(model, epochs, loss_fun, optimizer, X_train, y_train):
    for n in range(epochs):
        for x,y in zip(X_train, y_train):
            x_tensor = torch.transpose(torch.tensor(x),0,1)
            y_tensor = torch.tensor(y)
            y_pred = model(x_tensor)
            loss = loss_fun(y_pred, torch.transpose(y_tensor,0,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

# Evaluate the model performance on the training set
def eval_model_train(model, X_test, y_test):
    prediction = []
    for x,y in zip(X_test, y_test): 
        x_tensor = torch.transpose(torch.tensor(x),0,1)
        prediction.append(torch.argmax(model(x_tensor)).tolist())
    correct = 0
    print(prediction[0:50])
    print(y_test[0:50])
    for i in range(len(prediction)):
        if (prediction[i] == np.argmax(y_test[i])):
            correct += 1
    acc = float(correct/len(prediction))
    return acc

#Evaluate the model performance on the testing set
def eval_model(model, X_test, y_test):
    prediction = []
    incorrect = []
    for x,y in zip(X_test, y_test): 
        x_tensor = torch.transpose(torch.tensor(x),0,1)
        prediction.append(torch.argmax(model(x_tensor)).tolist())
    correct = 0
    for i in range(len(prediction)):
        if (prediction[i] == y_test[i]):
            correct += 1
        else:
            incorrect.append(i)
    print(incorrect)
    acc = float(correct/len(prediction))
    return acc