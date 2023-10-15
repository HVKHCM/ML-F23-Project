#All models are from sklearn:
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html and 
# SVM link here

#TODO Write a function to graph and record (and store in a file) accuracy as the model is training AND as the model is testing
#TODO Run the model on multiple different amounts of data for the above and with different hyperparameters

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

#Read data from csv

def get_data(data_csv_path:str): #https://pandas.pydata.org/pandas-docs/stable/reference/frame.html 
    df = pd.read_csv(data_csv_path)
    df.columns=["x"+str(i) for i in range(len(df.columns))]
    df.rename(columns = {list(df)[len(df.columns)-1]:'label'}, inplace=True)
    labels_list = []
    labels_df = df["label"]

    for label in labels_df:
        labels_list.append(label)

    labels = np.array(labels_list)

    df_without_labels = df.drop("label", axis=1)

    return df_without_labels, labels
        




dataset1_X_train, dataset1_y_train = get_data("../dataset1_training_data.csv")
dataset1_X_val, dataset1_y_val = get_data("../dataset1_validation_data.csv")
dataset1_X_test, dataset2_y_test = get_data("../dataset1_testing_data.csv")

dataset2_X_train, dataset2_y_train = get_data("../dataset2_training_data.csv")
dataset2_X_val, dataset2_y_val = get_data("../dataset2_validation_data.csv")
dataset2_X_test, dataset2_y_test = get_data("../dataset2_testing_data.csv")

penalty_type = 'l2' #only l2 or no regularization for the default solver. Change solvers?
lr = LogisticRegression(penalty=penalty_type)
model = lr.fit(X, y)






