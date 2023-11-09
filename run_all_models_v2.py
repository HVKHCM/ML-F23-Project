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

#Get data
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

#Process data
def process_data():

    dataset1_X_train, dataset1_y_train = get_data("dataset1_training_data.csv")
    dataset1_X_val, dataset1_y_val = get_data("dataset1_testing_data.csv")

    dataset2_X_train, dataset2_y_train = get_data("dataset2_training_data.csv")
    dataset2_X_val, dataset2_y_val = get_data("dataset2_testing_data.csv")


    #create heatmap

    fig, ax =plt.subplots(ncols=2, figsize=(20, 8))
    ax[0].set_title('Correlation data 1', fontsize=30)
    ax[1].set_title('Correlation data 2', fontsize=30)

    sns.heatmap(dataset1_X_train.corr(), ax=ax[0])
    sns.heatmap(dataset2_X_train.corr(), ax=ax[1])

    plt.show()

    #Remove high correlated features from dataset (training and test)

    # Create correlation matrix
    corr_matrix_1 = dataset1_X_train.corr()
    
    corr_matrix_2 = dataset2_X_train.corr()

    # Select upper triangle of correlation matrix
    upper_1= corr_matrix_1.where(np.triu(np.ones(corr_matrix_1.shape), k=1).astype(np.bool_))
    
    upper_2= corr_matrix_2.where(np.triu(np.ones(corr_matrix_2.shape), k=1).astype(np.bool_))

    # Find index of feature columns with correlation greater than 0.95
    to_drop_1 = [column for column in upper_1.columns if any(upper_1[column] > 0.95)]
    
    to_drop_2 = [column for column in upper_2.columns if any(upper_2[column] > 0.95)]

    #new Data Frame without high correlated featues
    dataset1_X_train=dataset1_X_train.drop(dataset1_X_train[to_drop_1], axis=1)
    dataset1_X_val=dataset1_X_val.drop(dataset1_X_val[to_drop_1], axis=1)
    
    dataset2_X_train=dataset2_X_train.drop(dataset2_X_train[to_drop_2], axis=1)
    dataset2_X_val=dataset2_X_val.drop(dataset2_X_val[to_drop_2], axis=1) 
    
    return dataset1_X_train, dataset1_y_train, dataset1_X_val, dataset1_y_val, dataset2_X_train, dataset2_y_train, dataset2_X_val, dataset2_y_val 

def test_model(model, predictions, X_val, y_val):

    accuracy = accuracy_score(y_val, predictions)
    
    precision = precision_score(y_val, predictions)
    
    recall = recall_score(y_val, predictions)
    
    f1 = f1_score(y_val, predictions)
    
    roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    
    return accuracy, precision, recall, f1, roc_auc

def calculate_best_models(models_dict:dict, output_csv_path:str):
    output_csv_file = open(output_csv_path, 'w', newline='')
    output_csv_writer = csv.writer(output_csv_file)

    best_accuracy = 0
    best_accuracy_model = ""

    best_precision = 0
    best_precision_model = ""

    best_recall = 0
    best_recall_model = ""

    best_f1 = 0
    best_f1_model = ""

    best_roc_auc = 0
    best_roc_auc_model = ""

    model_names = models_dict.keys()

    for model_name in model_names:

        #dictionary values are evaluation metrics for each model in a list of the form [acccuracy, precision, recall, f1, roc_auc]

        accuracy = models_dict[model_name][0]
        precision = models_dict[model_name][1]
        recall = models_dict[model_name][2]
        f1 = models_dict[model_name][3]
        roc_auc = models_dict[model_name][4]

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_accuracy_model = model_name

        if precision > best_precision:
            best_precision = precision
            best_precision_model = model_name

        if recall > best_recall:
            best_recall = recall
            best_recall_model = model_name

        if f1 > best_f1:
            best_f1 = f1
            best_f1_model = model_name

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_roc_auc_model = model_name

    output_csv_writer.writerow(["Model", "[Accuracy, Precision, Recall, F-1, ROC_AUC]", "Description"])
    output_csv_writer.writerow([best_accuracy_model, models_dict[best_accuracy_model], "Model with highest accuracy"])
    output_csv_writer.writerow([best_precision_model, models_dict[best_precision_model], "Model with highest precision"])
    output_csv_writer.writerow([best_recall_model, models_dict[best_recall_model], "Model with highest recall"])
    output_csv_writer.writerow([best_f1_model, models_dict[best_f1_model], "Model with highest f-1 score"])
    output_csv_writer.writerow([best_roc_auc_model, models_dict[best_roc_auc_model], "Model with highest roc-auc"])

    return best_accuracy_model


#Decision Tree
def train_and_test_decision_tree(X_train, y_train, X_val, y_val, output_csv_path):

    models_dict = {}

    for criterion in ["gini"]:
        for max_depth in range(3,8):
            for min_samples_split in range(2,5):
                for min_samples_leaf in range(1,5):

                    dt_model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                    dt_model_trained = dt_model.fit(X_train, y_train)
    
                    predictions = []
    
                    for test_example in X_val.iterrows(): #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows
                        test_example_df = pd.DataFrame(test_example[1]).T #get each row of the df separately
                        prediction = dt_model_trained.predict(test_example_df)[0]
                        predictions.append(prediction)

                    accuracy, precision, recall, f1, roc_auc = test_model(dt_model_trained, predictions, X_val, y_val)

                    models_dict[dt_model_trained] = [accuracy, precision, recall, f1, roc_auc]

    dt_with_highest_accuracy = calculate_best_models(models_dict, output_csv_path)

    print("Done running DecisionTreeClassifier!")

    return dt_with_highest_accuracy

#Boosting
def train_and_test_boosting(dt, X_train, y_train, X_val, y_val, output_csv_path):

    models_dict = {}
    
    bag_clf = BaggingClassifier(
    dt, n_estimators=500,
    max_samples=0.7, bootstrap=True, n_jobs=-1)  #Sampled with replecement. Each sample is 70% of the whole data. 500 DT trained
    
    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_val)
    
    num_test_examples = len(X_val)
    num_test_labels = len(y_val)

    assert num_test_examples == num_test_labels
    
    predictions = []
    
    for test_example in X_val.iterrows(): #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows
        test_example_df = pd.DataFrame(test_example[1]).T #get each row of the df separately
        prediction = bag_clf.predict(test_example_df)[0]
        predictions.append(prediction)
       
    accuracy, precision, recall, f1, roc_auc = test_model(bag_clf, predictions, X_val, y_val)
    models_dict[bag_clf] = [accuracy, precision, recall, f1, roc_auc]

    calculate_best_models(models_dict, output_csv_path)

    print("Done running BaggingClassifier!")

#Logistic Regression
def train_and_test_logistic_regression(X_train, y_train, X_val, y_val, output_csv_path):

    models_dict = {}

    for penalty in ["l1", "l2"]:
        for solver in ["liblinear"]:
            for max_iter in [(i * 100) for i in range (1, 11)]:

                lr_model = LogisticRegression(penalty=penalty, solver=solver, max_iter=max_iter)
                lr_model_trained = lr_model.fit(X_train, y_train)

                predictions = []

                for test_example in X_val.iterrows(): #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows
                    test_example_df = pd.DataFrame(test_example[1]).T #get each row of the df separately
                    prediction = lr_model_trained.predict(test_example_df)[0]
                    predictions.append(prediction)

                accuracy, precision, recall, f1, roc_auc = test_model(lr_model_trained, predictions, X_val, y_val)

                models_dict[lr_model_trained] = [accuracy, precision, recall, f1, roc_auc]

    calculate_best_models(models_dict, output_csv_path)

    print("Done running LogisticRegression!")

    
#SVM
def train_and_test_svm(X_train, y_train, X_val, y_val, output_csv_path):
    
    models_dict = {}

    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        for kernel in ["rbf", "linear", "poly", "sigmoid"]:
            for degree in [1,2,3,4,5]:
                for gamma in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:

                    svm_model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, probability=True)
                    svm_model_trained = svm_model.fit(X_train, y_train)
    
                    predictions = []
    
                    for test_example in X_val.iterrows(): #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows
                        test_example_df = pd.DataFrame(test_example[1]).T #get each row of the df separately
                        prediction = svm_model_trained.predict(test_example_df)[0]
                        predictions.append(prediction)

                    accuracy, precision, recall, f1, roc_auc = test_model(svm_model_trained, predictions, X_val, y_val)

                    models_dict[svm_model_trained] = [accuracy, precision, recall, f1, roc_auc]

    calculate_best_models(models_dict, output_csv_path)

    print("Done running SVM!")

#KNN
def train_and_test_kfold_knn (X_train, y_train, X_val, y_val, output_csv_path, range_tune=[1,10],fold=10):
    #create new a knn model
    assert range_tune [0] < range_tune[1]
    knn = KNeighborsClassifier()
    #create a dictionary of all values we want to test for n_neighbors
    param_grid = {'n_neighbors': np.arange(range_tune[0], range_tune[1])}
    #use gridsearch to test all values for n_neighbors
    knn_gsf = GridSearchCV(knn, param_grid, cv=fold, refit=True)
    #fit model to data
    knn_gsf.fit(X_train, y_train)

    best_parameters = knn_gsf.best_params_
    best_knn = knn_gsf.best_estimator_
    
    num_test_examples = len(X_val)
    num_test_labels = len(y_val)

    assert num_test_examples == num_test_labels
    
    predictions = []

    
    for test_example in X_val.iterrows(): #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows
        test_example_df = pd.DataFrame(test_example[1]).T #get each row of the df separately
        prediction = best_knn.predict(test_example_df)[0]
        predictions.append(prediction)

    test_model(best_knn, predictions, X_val, y_val, output_csv_path)
    

#Random Forests
def train_and_test_kfold_random_forest(X_train, y_train, X_val, y_val, output_csv_path, trees = [1,10], depth=[1,10], fold=10):
    assert trees[0] < trees[1]
    assert depth[0] < depth[1]
    rf = RandomForestClassifier()
    param_grid = {'n_estimators': np.arange(trees[0], trees[1]), 'max_depth': np.arange(depth[0],depth[1])}
    rf_gsf = GridSearchCV(rf, param_grid, cv=fold, refit=True)

    rf_gsf.fit(X_train,y_train)

    best_parameters = rf_gsf.best_params_
    best_rf = rf_gsf.best_estimator_
    
    num_test_examples = len(X_val)
    num_test_labels = len(y_val)

    assert num_test_examples == num_test_labels
    
    predictions = []
    
    for test_example in X_val.iterrows(): #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows
        test_example_df = pd.DataFrame(test_example[1]).T #get each row of the df separately
        prediction = best_rf.predict(test_example_df)[0]
        predictions.append(prediction)

    test_model(best_rf, predictions, X_val, y_val, output_csv_path)



dataset1_X_train, dataset1_y_train, dataset1_X_val, dataset1_y_val, dataset2_X_train, dataset2_y_train, dataset2_X_val, dataset2_y_val = process_data()

# dt_dataset1 = train_and_test_decision_tree(dataset1_X_train, dataset1_y_train, dataset1_X_val, dataset1_y_val, "output_metrics/decision_tree_dataset1.csv")
# dt_dataset2 = train_and_test_decision_tree(dataset2_X_train, dataset2_y_train, dataset2_X_val, dataset2_y_val, "output_metrics/decision_tree_dataset2.csv")

# train_and_test_boosting(dt_dataset1, dataset1_X_train, dataset1_y_train, dataset1_X_val, dataset1_y_val, "output_metrics/boosting_dataset1.csv")
# train_and_test_boosting(dt_dataset2, dataset2_X_train, dataset2_y_train, dataset2_X_val, dataset2_y_val, "output_metrics/boosting_dataset2.csv")

# train_and_test_logistic_regression(dataset1_X_train, dataset1_y_train, dataset1_X_val, dataset1_y_val, "output_metrics/logistic_regression_dataset1.csv")
# train_and_test_logistic_regression(dataset2_X_train, dataset2_y_train, dataset2_X_val, dataset2_y_val, "output_metrics/logistic_regression_dataset2.csv")

train_and_test_svm(dataset1_X_train, dataset1_y_train, dataset1_X_val, dataset1_y_val, "output_metrics/svm_dataset1.csv")

#TODO come back to this one
#train_and_test_svm(dataset2_X_train, dataset2_y_train, dataset2_X_val, dataset2_y_val, "output_metrics/svm_dataset2.csv")

#TODO come back to these 2
#train_and_test_kfold_knn(dataset1_X_train, dataset1_y_train, dataset1_X_val, dataset1_y_val, "output_metrics/kfold_knn_dataset1.csv")
#train_and_test_kfold_knn(dataset2_X_train, dataset2_y_train, dataset2_X_val, dataset2_y_val, "output_metrics/kfold_knn_dataset2.csv")

# train_and_test_kfold_random_forest(dataset1_X_train, dataset1_y_train, dataset1_X_val, dataset1_y_val, "output_metrics/kfold_random_forest_dataset1.csv")
# train_and_test_kfold_random_forest(dataset2_X_train, dataset2_y_train, dataset2_X_val, dataset2_y_val, "output_metrics/kfold_random_forest_dataset2.csv")


