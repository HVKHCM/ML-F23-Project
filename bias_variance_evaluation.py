#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,roc_auc_score,roc_curve
from sklearn.model_selection import cross_val_score,cross_val_predict, GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import csv
from run_all_models_v3 import get_data, process_data, test_model, get_train_and_test_predictions

#Decision Tree
def bias_variance_decision_tree(X_train, y_train, X_test, y_test, output_csv_path):

    output_csv_file = open(output_csv_path, 'w', newline='')
    writer = csv.writer(output_csv_file)
    writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Train Error", "Test Error", "max_depth", "min_samples_split", "min_samples_leaf"])

    for criterion in ["gini"]:
        for max_depth in range(1,9): 
            for min_samples_split in range(2,5): 
                for min_samples_leaf in range(1,5): 

                    dt_model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                    dt_model_trained = dt_model.fit(X_train, y_train)

                    test_row = test_model(dt_model_trained, X_train, y_train, X_test, y_test)
                    test_row = test_row + [max_depth, min_samples_split, min_samples_leaf]

                    writer.writerow(test_row)


    print("Done running DecisionTreeClassifier!")

#Boosting
def bias_variance_boosting(dt, X_train, y_train, X_test, y_test, output_csv_path):

    output_csv_file = open(output_csv_path, 'w', newline='')
    writer = csv.writer(output_csv_file)
    writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Train Error", "Test Error", "n_estimators", "max_samples"])

    for n_estimators in [100,200, 300, 400, 500, 600, 700]:
        for max_samples in [0.5,0.6,0.7,0.8, 0.9, 1.0]: 
            bag_model = BaggingClassifier( dt, n_estimators=n_estimators,max_samples=max_samples, bootstrap=True, n_jobs=-1)
            bag_model_trained = bag_model.fit(X_train, y_train)

            test_row = test_model(bag_model_trained, X_train, y_train, X_test, y_test)
            test_row = test_row + [n_estimators, max_samples]

            writer.writerow(test_row)


    print("Done running BaggingClassifier!")

#Logistic Regression
def bias_variance_logistic_regression(X_train, y_train, X_test, y_test, output_csv_path):

    output_csv_file = open(output_csv_path, 'w', newline='')
    writer = csv.writer(output_csv_file)
    writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Train Error", "Test Error", "penalty", "C", "max_iter"])

    models_dict = {}

    for penalty in ["l1", "l2"]:
        for C in [1.0, 10, 100, 1000, 10000]:
            for solver in ["liblinear"]:
                for max_iter in [(i * 100) for i in range (1, 11)]:

                    lr_model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter)
                    lr_model_trained = lr_model.fit(X_train, y_train)

                    test_row = test_model(lr_model_trained, X_train, y_train, X_test, y_test)
                    test_row = test_row + [penalty, C, max_iter]

                    writer.writerow(test_row)


    print("Done running LogisticRegression!")

    
#SVM
def bias_variance_svm(X_train, y_train, X_test, y_test, output_csv_path):
    
    output_csv_file = open(output_csv_path, 'w', newline='')
    writer = csv.writer(output_csv_file)
    writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Train Error", "Test Error", "C", "kernel", "degree", "gamma"])

    for C in [0.1, 1.0, 10.0]:
        for kernel in ["rbf", "linear", "sigmoid"]:
            #for degree in [2,3,4]:
            for gamma in [0.1, 1.0, 10.0]:

                svm_model = SVC(C=C, kernel=kernel,gamma=gamma, probability=True)
                svm_model_trained = svm_model.fit(X_train, y_train)
    
                test_row = test_model(svm_model_trained, X_train, y_train, X_test, y_test)
                test_row = test_row + [C, kernel, gamma]

                writer.writerow(test_row)

    print("Done running SVM!")

#KNN
def bias_variance_kfold_knn (X_train, y_train, X_test, y_test, output_csv_path, range_tune=[1,10],fold=10):
    assert range_tune [0] < range_tune[1]
    
    output_csv_file = open(output_csv_path, 'w', newline='')
    writer = csv.writer(output_csv_file)
    writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Train Error", "Test Error", "n_neighbors"])

    for n_neighbors in np.arange(range_tune[0], range_tune[1]):

        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn_model_trained = knn_model.fit(X_train, y_train)

        test_row = test_model(knn_model_trained, X_train, y_train, X_test, y_test)
        test_row = test_row + [n_neighbors]

        writer.writerow(test_row)

    print("Done running KNeighborsClassifier!")
    

#Random Forests
def bias_variance_kfold_random_forest(X_train, y_train, X_test, y_test, output_csv_path, trees = [1,10], depth=[1,10], fold=10):
    assert trees[0] < trees[1]
    assert depth[0] < depth[1]

    output_csv_file = open(output_csv_path, 'w', newline='')
    writer = csv.writer(output_csv_file)
    writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Train Error", "Test Error", "n_estimators", "max_depth"])

    for n_estimators in np.arange(trees[0], trees[1]):
        for max_depth in np.arange(depth[0],depth[1]):

            rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            rf_model_trained = rf_model.fit(X_train, y_train)

            test_row = test_model(rf_model_trained, X_train, y_train, X_test, y_test)
            test_row = test_row + [n_estimators, max_depth]

            writer.writerow(test_row)
            
    
    print("Done running RandomForestClassifier!")


dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test, dataset2_X_train, dataset2_y_train, dataset2_X_test, dataset2_y_test = process_data()

# dt_dataset1 = bias_variance_decision_tree(dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test, "bias_variance_comparisons/decision_tree_dataset1.csv")
# dt_dataset2 = bias_variance_decision_tree(dataset2_X_train, dataset2_y_train, dataset2_X_test, dataset2_y_test, "bias_variance_comparisons/decision_tree_dataset2.csv")

# bias_variance_boosting(DecisionTreeClassifier(max_depth=5), dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test, "bias_variance_comparisons/boosting_dataset1.csv")
# bias_variance_boosting(DecisionTreeClassifier(max_depth=3, min_samples_split=4), dataset2_X_train, dataset2_y_train, dataset2_X_test, dataset2_y_test, "bias_variance_comparisons/boosting_dataset2.csv")

# bias_variance_logistic_regression(dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test, "bias_variance_comparisons/logistic_regression_dataset1.csv")
# bias_variance_logistic_regression(dataset2_X_train, dataset2_y_train, dataset2_X_test, dataset2_y_test, "bias_variance_comparisons/logistic_regression_dataset2.csv")

#Caroline can't run these
bias_variance_svm(dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test, "bias_variance_comparisons/svm_dataset1.csv")
bias_variance_svm(dataset2_X_train, dataset2_y_train, dataset2_X_test, dataset2_y_test, "bias_variance_comparisons/svm_dataset2.csv")

#Caroline can't run these
#bias_variance_kfold_knn(dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test, "bias_variance_comparisons/kfold_knn_dataset1.csv")
#bias_variance_kfold_knn(dataset2_X_train, dataset2_y_train, dataset2_X_test, dataset2_y_test, "bias_variance_comparisons/kfold_knn_dataset2.csv")

# bias_variance_kfold_random_forest(dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test, "bias_variance_comparisons/kfold_random_forest_dataset1.csv")
# bias_variance_kfold_random_forest(dataset2_X_train, dataset2_y_train, dataset2_X_test, dataset2_y_test, "bias_variance_comparisons/kfold_random_forest_dataset2.csv")
