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

#model = utils.Net(400,100)
#num_epochs = np.arange(10,50,10)
#optimizer = optim.SGD(model.parameters(), lr=0.01)
#loss_fn = nn.CrossEntropyLoss()

#accuracy = []

#for e in num_epochs:
#    print(e)
#    trained_model = utils.train_model(model,e, loss_fn, optimizer, X_train, y_train)
#    accuracy.append(utils.eval_model(trained_model,X_test,y_test))
#    print(accuracy[-1])

#plt.scatter(num_epochs, accuracy)
#plt.show()



unit1 = np.arange(10, 101, 20)
unit2 = np.arange(10, 61, 20)
lr = [0.01,0.1,0.5,0.9]
epoch = 10

loss_fn = nn.CrossEntropyLoss()

train_error, accuracy_list,configuration_list = utils.unit_optimize(epochs=epoch, loss_fun=loss_fn,X_train=X_train, y_train=y_train, 
                                   X_test=X_test, y_test=y_test, unit1_range=unit1, unit2_range=unit2, lr_range=lr)

""" hidden1 = []
hidden2 = []
learning_rate = []
epochs = []

for i in range(len(configuration_list)):
    epochs.append(epoch)
    hidden1.append(configuration_list[i][0])
    hidden2.append(configuration_list[i][1])
    learning_rate.append(configuration_list[i][2])


data_bk = pd.DataFrame(list(zip(configuration_list, accuracy_list, train_error)),columns=['Configuration','Accuracy', 'Train Error'])
data_bk.to_csv("backup.csv", index=False)
data_real = pd.DataFrame(list(zip(epochs, hidden1,hidden2,learning_rate, accuracy_list, train_error)),columns=['Number of epochs','First Unit', 'Second Unit',
                                                                                                  'Learning Rate', 'Accuracy', 'Train Accuracy'])
data_real.to_csv("data.csv", index=False) """

#trained_model = utils.train_model(model, num_epochs, loss_fn, optimizer, X_train, y_train)

#acc = utils.eval_model(trained_model, X_test, y_test)
#for n in range(num_epochs):
#    count = 0
#    print("epoch: {}".format(n))
#    for x,y in zip(X_train, y_train):
#        x_tensor = torch.transpose(torch.tensor(x),0,1)
#        y_tensor = torch.tensor(y)
#       y_pred = model(x_tensor)
#       loss = loss_fn(y_pred, torch.transpose(y_tensor,0,1))
#        optimizer.zero_grad()
#       loss.backward()
#       optimizer.step()

#model.eval()
#prediction = []
#for x,y in zip(X_test, y_test): 
#    x_tensor = torch.transpose(torch.tensor(x),0,1)
#    prediction.append(torch.argmax(model(x_tensor)).tolist())
#correct = 0
#for i in range(len(prediction)):
#    if (prediction[i] == y_test[i]):
#       correct += 1
#acc = float(correct/len(prediction))
#print("Model accuracy: %.2f%%" % (acc*100))
#        count += 1
#print(prediction)