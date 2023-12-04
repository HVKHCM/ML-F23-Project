import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mlxtend.evaluate import bias_variance_decomp
import pandas as pd
import numpy as np

def get_bias_variance(model, X_train, y_train, X_test, y_test):
    X_train_for_bv = pd.DataFrame.from_records(X_train)
    y_train_for_bv = pd.DataFrame.from_records(y_train)
    X_test_for_bv = pd.DataFrame.from_records(X_test)
    y_test_for_bv = pd.DataFrame(y_test)
    X_train = X_train_for_bv.values
    y_train = y_train_for_bv.values
    X_test = X_test_for_bv.values
    y_test = y_test_for_bv.values
    _, bias, var = bias_variance_decomp(model, X_train, y_train, X_test, y_test, '0-1_loss', random_seed=1)
    return bias, var



def generate_model():
    model = nn.Sequential(
        nn.Linear(784,100),
        nn.Sigmoid(),
        nn.Linear(100, 50),
        nn.Sigmoid(),
        nn.Linear(50, 20),
        nn.Sigmoid(),
        nn.Linear(20, 10),
        nn.Softmax()
    )
    return model


class Net(nn.Module):
    def __init__(self, unit1, unit2, unit3):
        super().__init__()
        self.layers = nn.ModuleDict()

        self.layers["input"] = nn.Linear(in_features=784, out_features=unit1)
        
        self.layers["hidden_0"] = nn.Linear(in_features=unit1, out_features=unit2)

        self.layers["hidden_1"] = nn.Linear(in_features=unit2, out_features=unit3)
        
        self.layers["output"] = nn.Linear(in_features=unit3, out_features=10)

    def forward(self,x):
        x = self.layers["input"](x)

        for i in range(2):
            x = F.sigmoid(self.layers[f"hidden_{i}"](x))
        
        return F.softmax(self.layers["output"](x))


def unit_optimize(epochs, loss_fun, X_train, y_train, X_test, y_test, unit1_range, unit2_range, unit3_range, lr_range):
    accuracy_list = []
    #bias_list = []
    #var_list = []
    all_config = []
    train_error = []
    count = 0
    for i in unit1_range:
        for j in unit2_range:
            for k in unit3_range:
                for l in lr_range:
                    print("Config: {}".format(count + 1))
                    config = []
                    config.append(i)
                    config.append(j)
                    config.append(k)
                    config.append(l)
                    model = Net(i, j, k)
                    optimizer = optim.SGD(model.parameters(), lr=l)
                    trained_model = train_model(model,epochs, loss_fun, optimizer, X_train, y_train)
                    accuracy_list.append(eval_model(trained_model, X_test, y_test))
                    train_error.append(eval_model_train(trained_model, X_train, y_train))
                    #bias, variance = get_bias_variance(model, X_train, y_train, X_test, y_test)
                    #bias_list.append(bias)
                    #var_list.append(variance)
                    all_config.append(config)
                    count += 1
    return train_error, accuracy_list, all_config

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

def eval_model_train(model, X_test, y_test):
    model.eval()
    prediction = []
    for x,y in zip(X_test, y_test): 
        x_tensor = torch.transpose(torch.tensor(x),0,1)
        prediction.append(torch.argmax(model(x_tensor)).tolist())
    correct = 0
    for i in range(len(prediction)):
        if (prediction[i] == np.argmax(y_test[i])):
            correct += 1
    acc = float(correct/len(prediction))
    return acc

def eval_model(model, X_test, y_test):
    model.eval()
    prediction = []
    for x,y in zip(X_test, y_test): 
        x_tensor = torch.transpose(torch.tensor(x),0,1)
        prediction.append(torch.argmax(model(x_tensor)).tolist())
    correct = 0
    for i in range(len(prediction)):
        if (prediction[i] == y_test[i]):
            correct += 1
    acc = float(correct/len(prediction))
    return acc