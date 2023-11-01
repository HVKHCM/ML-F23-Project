import torch 
import torch.nn as nn
import torch.nn.functional as F

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

def train_model(model, epochs, loss_fun, optimizer, X_train, y_train):
    for n in range(epochs):
        print("epoch: {}".format(n+1))
        for x,y in zip(X_train, y_train):
            x_tensor = torch.transpose(torch.tensor(x),0,1)
            y_tensor = torch.tensor(y)
            y_pred = model(x_tensor)
            loss = loss_fun(y_pred, torch.transpose(y_tensor,0,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

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