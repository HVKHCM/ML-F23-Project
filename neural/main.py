import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
import mnist_loader
import pandas as pd
import torch.optim as optim

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


unit1 = np.arange(10, 120, 15)
unit2 = np.arange(10, 60, 20)
unit3 = np.arange(10, 30, 20)
lr = np.arange(0.01,1,0.1)
epoch = 10



#print(y_test)

#X_train_tensor = torch.utils.data.DataLoader(X_df,batch_size=1,shuffle=True)

#model = utils.Net(100, 50, 20)
loss_fn = nn.CrossEntropyLoss()
#num_epochs = 2
#optimizer = optim.SGD(model.parameters(), lr=0.1)

accuracy_list, bias_list, var_list, configuration_list = utils.unit_optimize(epochs=epoch, loss_fun=loss_fn,X_train=X_train, y_train=y_train, 
                                   X_test=X_test, y_test=y_test, unit1_range=unit1, unit2_range=unit2, unit3_range=unit3, lr_range=lr)

hidden1 = []
hidden2 = []
hidden3 = []
learning_rate = []
epochs = []

for i in range(len(configuration_list)):
    epochs.append(epoch)
    hidden1.append(configuration_list[i][0])
    hidden2.append(configuration_list[i][1])
    hidden3.append(configuration_list[i][2])
    learning_rate.append(configuration_list[i][3])


data_bk = pd.DataFrame(list(zip(configuration_list, accuracy_list, bias_list, var_list)),columns=['Configuration','Accuracy', 'Bias', 'Variance'])
data_bk.to_csv("backup.csv", index=False)
data_real = pd.DataFrame(list(zip(epochs, hidden1,hidden2,hidden3,learning_rate, accuracy_list, bias_list, var_list)),columns=['Number of epochs','First Unit', 'Second Unit', 'Third Unit', 
                                                                                                  'Learning Rate', 'Accuracy', 'Bias', 'Variance'])
data_real.to_csv("data.csv", index=False)

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