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
    dataset1_X_test, dataset1_y_test = get_data("dataset1_testing_data.csv")

    dataset2_X_train, dataset2_y_train = get_data("dataset2_training_data.csv")
    dataset2_X_test, dataset2_y_test = get_data("dataset2_testing_data.csv")


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
    dataset1_X_test=dataset1_X_test.drop(dataset1_X_test[to_drop_1], axis=1)
    
    dataset2_X_train=dataset2_X_train.drop(dataset2_X_train[to_drop_2], axis=1)
    dataset2_X_test=dataset2_X_test.drop(dataset2_X_test[to_drop_2], axis=1) 
    
    return dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test, dataset2_X_train, dataset2_y_train, dataset2_X_test, dataset2_y_test 

def test_model(model, predictions, X_test, y_test, output_csv_path):

    output_csv_file = open(output_csv_path, 'w', newline='')
    writer = csv.writer(output_csv_file)
    writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F-1", "ROC AUC"])

    accuracy = accuracy_score(y_test, predictions)
    
    precision = precision_score(y_test, predictions)
    
    recall = recall_score(y_test, predictions)
    
    f1 = f1_score(y_test, predictions)
    
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    writer.writerow([model, accuracy, precision, recall, f1, roc_auc])
    
    print('accuracy')
    print(accuracy, '\n')
    
    print('precision')
    print(precision, '\n')
    
    print('recall')
    print(recall, '\n')

    print('f1')
    print(f1, '\n')
    
    print('roc auc')
    print(roc_auc, '\n')


#Decision Tree
def train_and_test_decision_tree(X_train, y_train, X_test, y_test, output_csv_path):
    
    gsc = GridSearchCV(estimator=DecisionTreeClassifier(criterion='gini'),param_grid={'max_depth': range(3,8), 'min_samples_split': range(2,5),'min_samples_leaf': range(1,5)},cv=10, scoring='accuracy', verbose=0, n_jobs=-1)    
    grid_result = gsc.fit(X_train, y_train) #Results of the Decision Tree classifier with the optimize hyperparameters after 10-fold CV
    dt = grid_result.best_estimator_
    best_params = grid_result.best_params_ #Best paramerts
    best_score = grid_result.best_score_ #Score of the best model

    model = dt.fit(X_train, y_train) 
    
    num_test_examples = len(X_test)
    num_test_labels = len(y_test)

    assert num_test_examples == num_test_labels
    
    predictions = []
    
    for test_example in X_test.iterrows(): #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows
        test_example_df = pd.DataFrame(test_example[1]).T #get each row of the df separately
        prediction = model.predict(test_example_df)[0]
        predictions.append(prediction)

    test_model(model, predictions, X_test, y_test, output_csv_path)

    return dt

#Boosting
def train_and_test_boosting(dt, X_train, y_train, X_test, y_test, output_csv_path):
    
    bag_clf = BaggingClassifier(
    dt, n_estimators=500,
    max_samples=0.7, bootstrap=True, n_jobs=-1)  #Sampled with replecement. Each sample is 70% of the whole data. 500 DT trained
    
    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)
    
    num_test_examples = len(X_test)
    num_test_labels = len(y_test)

    assert num_test_examples == num_test_labels
    
    predictions = []
    
    for test_example in X_test.iterrows(): #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows
        test_example_df = pd.DataFrame(test_example[1]).T #get each row of the df separately
        prediction = bag_clf.predict(test_example_df)[0]
        predictions.append(prediction)
       
    test_model(bag_clf, predictions, X_test, y_test, output_csv_path)

#Logistic Regression
def train_and_test_logistic_regression(X_train, y_train, X_test, y_test, output_csv_path):
    
    parameters = {
        "penalty" : ("l1", "l2"),
        "max_iter" : [(i * 100) for i in range (1, 11)]
    }

    gsc = GridSearchCV(estimator=LogisticRegression(solver="liblinear"), param_grid=parameters, cv=10, scoring="accuracy") #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    grid_result = gsc.fit(X_train, y_train) #TODO accuracy?
    best_params = grid_result.best_params_
    best_lr = grid_result.best_estimator_
    best_score = grid_result.best_score_
    
    lr_model = best_lr.fit(X_train, y_train)

    num_test_examples = len(X_test)
    num_test_labels = len(y_test)

    assert num_test_examples == num_test_labels
    
    predictions = []
    
    for test_example in X_test.iterrows(): #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows
        test_example_df = pd.DataFrame(test_example[1]).T #get each row of the df separately
        prediction = lr_model.predict(test_example_df)[0]
        predictions.append(prediction)

    test_model(lr_model, predictions, X_test, y_test, output_csv_path)
    
#SVM
def train_and_test_svm(X_train, y_train, X_test, y_test, output_csv_path):
    
    parameters = {
        "C" : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "kernel" : ["rbf", "linear", "poly", "sigmoid"],
        "degree" : [1,2,3,4,5],
        "gamma" : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }

    gsc = GridSearchCV(estimator=SVC(probability=True), param_grid=parameters, cv=10, scoring="accuracy") #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    grid_result = gsc.fit(X_train, y_train) #TODO accuracy?
    best_params = grid_result.best_params_
    best_svm = grid_result.best_estimator_
    best_score = grid_result.best_score_
    
    svm_model = best_svm.fit(X_train, y_train)

    num_test_examples = len(X_test)
    num_test_labels = len(y_test)

    assert num_test_examples == num_test_labels
    
    predictions = []
    
    for test_example in X_test.iterrows(): #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows
        test_example_df = pd.DataFrame(test_example[1]).T #get each row of the df separately
        prediction = svm_model.predict(test_example_df)[0]
        predictions.append(prediction)

    test_model(svm_model, predictions, X_test, y_test, output_csv_path)

#KNN
def train_and_test_kfold_knn (X_train, y_train, X_test, y_test, output_csv_path, range_tune=[1,10],fold=10):
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
    
    num_test_examples = len(X_test)
    num_test_labels = len(y_test)

    assert num_test_examples == num_test_labels
    
    predictions = []

    
    for test_example in X_test.iterrows(): #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows
        test_example_df = pd.DataFrame(test_example[1]).T #get each row of the df separately
        prediction = best_knn.predict(test_example_df)[0]
        predictions.append(prediction)

    test_model(best_knn, predictions, X_test, y_test, output_csv_path)
    

#Random Forests
def train_and_test_kfold_random_forest(X_train, y_train, X_test, y_test, output_csv_path, trees = [1,10], depth=[1,10], fold=10):
    assert trees[0] < trees[1]
    assert depth[0] < depth[1]
    rf = RandomForestClassifier()
    param_grid = {'n_estimators': np.arange(trees[0], trees[1]), 'max_depth': np.arange(depth[0],depth[1])}
    rf_gsf = GridSearchCV(rf, param_grid, cv=fold, refit=True)

    rf_gsf.fit(X_train,y_train)

    best_parameters = rf_gsf.best_params_
    best_rf = rf_gsf.best_estimator_
    
    num_test_examples = len(X_test)
    num_test_labels = len(y_test)

    assert num_test_examples == num_test_labels
    
    predictions = []
    
    for test_example in X_test.iterrows(): #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows
        test_example_df = pd.DataFrame(test_example[1]).T #get each row of the df separately
        prediction = best_rf.predict(test_example_df)[0]
        predictions.append(prediction)

    test_model(best_rf, predictions, X_test, y_test, output_csv_path)



dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test, dataset2_X_train, dataset2_y_train, dataset2_X_test, dataset2_y_test = process_data()

# dt_dataset1 = train_and_test_decision_tree(dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test, "output_metrics/decision_tree_dataset1.csv")
# dt_dataset2 = train_and_test_decision_tree(dataset2_X_train, dataset2_y_train, dataset2_X_test, dataset2_y_test, "output_metrics/decision_tree_dataset2.csv")

# train_and_test_boosting(dt_dataset1, dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test, "output_metrics/boosting_dataset1.csv")
# train_and_test_boosting(dt_dataset2, dataset2_X_train, dataset2_y_train, dataset2_X_test, dataset2_y_test, "output_metrics/boosting_dataset2.csv")

# train_and_test_logistic_regression(dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test, "output_metrics/logistic_regression_dataset1.csv")
# train_and_test_logistic_regression(dataset2_X_train, dataset2_y_train, dataset2_X_test, dataset2_y_test, "output_metrics/logistic_regression_dataset2.csv")

# train_and_test_svm(dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test, "output_metrics/svm_dataset1.csv")

#TODO come back to this one
train_and_test_svm(dataset2_X_train, dataset2_y_train, dataset2_X_test, dataset2_y_test, "output_metrics/svm_dataset2.csv")

#TODO come back to these 2
#train_and_test_kfold_knn(dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test, "output_metrics/kfold_knn_dataset1.csv")
#train_and_test_kfold_knn(dataset2_X_train, dataset2_y_train, dataset2_X_test, dataset2_y_test, "output_metrics/kfold_knn_dataset2.csv")

# train_and_test_kfold_random_forest(dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test, "output_metrics/kfold_random_forest_dataset1.csv")
# train_and_test_kfold_random_forest(dataset2_X_train, dataset2_y_train, dataset2_X_test, dataset2_y_test, "output_metrics/kfold_random_forest_dataset2.csv")


