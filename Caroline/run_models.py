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
#dataset1_X_val, dataset1_y_val = get_data("../dataset1_validation_data.csv")
dataset1_X_test, dataset2_y_test = get_data("../dataset1_testing_data.csv")

dataset2_X_train, dataset2_y_train = get_data("../dataset2_training_data.csv")
#dataset2_X_val, dataset2_y_val = get_data("../dataset2_validation_data.csv")
dataset2_X_test, dataset2_y_test = get_data("../dataset2_testing_data.csv")

#Run Logistic Regression
penalty_type = 'l2' #only l2 or no regularization for the default solver. Change solvers?
solver = 'lbfgs'
max_iterations = 500

def train_and_test_logistic_regression(penalty_type, solver, max_iterations, X_train, y_train, X_test, y_test):
    lr = LogisticRegression(penalty=penalty_type, solver=solver, max_iter=max_iterations)
    model = lr.fit(X_train, y_train)

    num_test_examples = len(X_test)
    num_test_labels = len(y_test)

    assert num_test_examples == num_test_labels

    predictions = []

    for test_example in X_test.iterrows(): #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows
        test_example_df = pd.DataFrame(test_example[1]).T #get each row of the df separately
        prediction = model.predict(test_example_df)[0]
        predictions.append(prediction)
    
    total_examples = 0
    num_true_positives = 0 #predicted class = true class = 1
    num_true_negatives = 0 #predicted class = true class = 0
    num_false_positives = 0 #predicted class = 1 and true class = 0
    num_false_negatives = 0 #predicted class = 0 and true class = 1

    for prediction, true_label in zip(predictions, y_test):
        total_examples += 1
        if prediction == 1 and true_label == 1:
            num_true_positives += 1
        elif prediction == 0 and true_label == 0:
            num_true_negatives += 1
        elif prediction == 1 and true_label == 0:
            num_false_positives += 1
        elif prediction == 0 and true_label == 1:
            num_false_negatives += 1
    
    accuracy = num_examples_correct/total_examples

    print(accuracy)

        
    # print(predictions)
    # print(y_test)

train_and_test_logistic_regression(penalty_type, solver, max_iterations, dataset2_X_train, dataset2_y_train, dataset2_X_test, dataset2_y_test)





