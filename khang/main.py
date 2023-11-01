import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
import utils

dataset1_X, dataset1_y = utils.get_data("../data1.csv")
X1_train, X1_test, y1_train, y1_test = train_test_split(dataset1_X, dataset1_y,random_state=12345, test_size=0.2, shuffle=True) 

dataset2_X, dataset2_y = utils.get_data("../data2.csv")
X2_train, X2_test, y2_train, y2_test = train_test_split(dataset2_X, dataset2_y,random_state=12345, test_size=0.2, shuffle=True) 



#best_knn_model, best_number_neighbors = utils.kfold_knn(X1_train, y1_train)
#print(best_number_neighbors ['n_neighbors'])
#precision, recall, fscore, auc = utils.test_and_evaluate_model(best_knn_model,X1_test, y1_test)


best_rf_model, best_params = utils.kfold_random_forest(X2_train, y2_train)
print(best_params['n_estimators'])
print(best_params['max_depth'])
precision, recall, fscore, auc = utils.test_and_evaluate_model(best_rf_model,X2_test, y2_test)
