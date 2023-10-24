#All models are from sklearn:
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html and 
#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

#TODO Write a function to graph and record (and store in a file) accuracy as the model is training AND as the model is testing
#TODO Run the model on multiple different amounts of data for the above and with different hyperparameters

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import csv

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

def test_and_evaluate_model(model, X_test, y_test):

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

    accuracy = 0
    precision = 0
    recall = 0
    f_1 = 0
    auc = 0

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

    print("True Positives:", num_true_positives)
    print("False Negatives:", num_false_negatives)
    print("False Positives:", num_false_positives)
    print("True Negatives:", num_true_negatives)

    #Calculate Accuracy
    if (num_true_positives + num_true_negatives + num_false_positives + num_false_negatives) != 0:
        accuracy = (num_true_positives + num_true_negatives) / (num_true_positives + num_true_negatives + num_false_positives + num_false_negatives)

    #Calculate Precision
    #fewer false positives is better
    if (num_true_positives + num_false_positives) != 0:
        precision = (num_true_positives) / (num_true_positives + num_false_positives)

    #Calculate Recall
    #fewer false negatives is better
    if (num_true_positives + num_false_negatives) != 0:
        recall = (num_true_positives) / (num_true_positives + num_false_negatives)

    #Calculate F-1
    #TODO verify that this is the correct metric
    if (recall + precision) != 0:
        f_1 = (2 * recall * precision) / (recall + precision)

    #Calculate AUC
    #TODO calculate AUC

    print("Model: ", model)
    print("\t", "Accuracy:", accuracy)
    print("\t", "Precision:", precision)
    print("\t", "Recall:", recall)
    print("\t", "F-1:", f_1)
    print("\n")

    return accuracy, precision, recall, f_1, auc

def train_and_test_logistic_regressions(penalties:list, solvers:list, max_iters:list, output_csv_path):
    output_csv_file = open(output_csv_path, 'w', newline='')
    output_csv_writer = csv.writer(output_csv_file)
    output_csv_writer.writerow(["Logistic Regression Model", "Dataset", "Penalty", "Solver", "Max Iterations", "Accuracy", "Precision", "Recall", "F-1", "AUC"])

    for penalty in penalties:
        for solver in solvers:
            for max_iter in max_iters:
                lr = LogisticRegression(penalty=penalty, solver=solver, max_iter=max_iter)
                lr_model_dataset1 = lr.fit(dataset1_X_train, dataset1_y_train)
                lr_model_dataset2 = lr.fit(dataset2_X_train, dataset2_y_train)
        
                accuracy_1, precision_1, recall_1, f_1_1, auc_1 = test_and_evaluate_model(lr_model_dataset1, dataset1_X_val, dataset1_y_val)
                #accuracy_2, precision_2, recall_2, f_1_2, auc_2 = test_and_evaluate_model(lr_model_dataset2, dataset2_X_val, dataset2_y_val)
                
                output_csv_writer.writerow([lr_model_dataset1, "1", penalty, solver, max_iter, accuracy_1, precision_1, recall_1, f_1_1, auc_1])
                #output_csv_writer.writerow([lr_model_dataset2, "2", penalty, solver, max_iter, accuracy_2, precision_2, recall_2, f_1_2, auc_2])
                
    print("Done training and testing Logistic Regression Models!")


def train_and_test_svms(Cs:list, kernels:list, degrees:list, gammas:list, output_csv_path):
    output_csv_file = open(output_csv_path, 'w', newline='')
    output_csv_writer = csv.writer(output_csv_file)
    output_csv_writer.writerow(["SVM Model", "Dataset", "C", "Kernel", "Degree", "Gamma", "Accuracy", "Precision", "Recall", "F-1", "AUC"])

    for C in Cs:
        for kernel in kernels:
            for degree in degrees:
                for gamma in gammas:
                    svm = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
                    svm_model_dataset1 = svm.fit(dataset1_X_train, dataset1_y_train)
                    svm_model_dataset2 = svm.fit(dataset2_X_train, dataset2_y_train)
                    print("SVM Dataset 2 Accuracy:", svm_model_dataset2.score(dataset2_X_val, dataset2_y_val))

                    #I'm not including the test set as an option here

                    #accuracy_1, precision_1, recall_1, f_1_1, auc_1 = test_and_evaluate_model(svm_model_dataset1, dataset1_X_val, dataset1_y_val)
                    accuracy_2, precision_2, recall_2, f_1_2, auc_2 = test_and_evaluate_model(svm_model_dataset2, dataset2_X_val, dataset2_y_val)

                    #output_csv_writer.writerow([svm_model_dataset1, "1", C, kernel, degree, gamma, accuracy_1, precision_1, recall_1, f_1_1, auc_1])
                    output_csv_writer.writerow([svm_model_dataset2, "2", C, kernel, degree, gamma, accuracy_2, precision_2, recall_2, f_1_2, auc_2])
    
    print("Done training and testing SVM Models!")




#Data
dataset1_X_train, dataset1_y_train = get_data("../dataset1_training_data.csv")
dataset1_X_val, dataset1_y_val = get_data("../dataset1_validation_data.csv")
dataset1_X_test, dataset1_y_test = get_data("../dataset1_testing_data.csv")

dataset2_X_train, dataset2_y_train = get_data("../dataset2_training_data.csv")
dataset2_X_val, dataset2_y_val = get_data("../dataset2_validation_data.csv")
dataset2_X_test, dataset2_y_test = get_data("../dataset2_testing_data.csv")
    
#Run Logistic Regression
# penalty_types = ['l2'] #only l2 or no regularization for the default solver. Change solvers?
# solvers = ['lbfgs']
# max_iterations = [2000]
# logistic_regression_validation_output_csv_path = "logistic_regression_validation_results.csv"
# train_and_test_logistic_regressions(penalty_types, solvers, max_iterations, logistic_regression_validation_output_csv_path)

#Run SVM
Cs = [1.0]
kernels = ['rbf']
degrees = [1]
gammas = [1.0]
svm_validation_output_csv_path = "svm_validation_results.csv"
train_and_test_svms(Cs, kernels, degrees, gammas, svm_validation_output_csv_path)





