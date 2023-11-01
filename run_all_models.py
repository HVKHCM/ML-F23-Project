#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score,roc_auc_score,roc_curve
from sklearn.model_selection import cross_val_score,cross_val_predict ,GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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

def test_model(model, predictions, X_test, y_test):

    accuracy = accuracy_score(y_test, predictions)
    
    precision = precision_score(y_test, predictions)
    
    recall = recall_score(y_test, predictions)
    
    f1 = f1_score(y_test, predictions)
    
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
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
def train_and_test_decision_tree(X_train, y_train, X_test, y_test):
    
    gsc = GridSearchCV(estimator=DecisionTreeClassifier(criterion='gini'),param_grid={'max_depth': range(3,8), 'min_samples_split': range(2,5),'min_samples_leaf': range(1,5)},cv=10, scoring='accuracy', verbose=0, n_jobs=-1)    
    grid_result = gsc.fit(X_train, y_train) #Results of the Decision Tree classifier with the optimize hyperparameters after 10-fold CV
    best_params = grid_result.best_params_ #Best paramerts
    best_score = grid_result.best_score_ #Score of the best model
    
    print("Best hyperparameters")
    print(best_params, '\n')
    
    print("Best score from CV")
    print(best_score, '\n')
    
    dt = DecisionTreeClassifier(criterion='gini',max_depth=best_params["max_depth"], random_state=0) #Create DT with optimal Hyperparameters

    model = dt.fit(X_train, y_train) 
    
    num_test_examples = len(X_test)
    num_test_labels = len(y_test)

    assert num_test_examples == num_test_labels
    
    predictions = []
    
    for test_example in X_test.iterrows(): #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows
        test_example_df = pd.DataFrame(test_example[1]).T #get each row of the df separately
        prediction = model.predict(test_example_df)[0]
        predictions.append(prediction)

    test_model(model, predictions, X_test, y_test)

    return dt

#Boosting
def train_and_test_boosting(dt, X_train, y_train, X_test, y_test):
    
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
       
    test_model(bag_clf, predictions, X_test, y_test)

#Logistic Regression
def train_and_test_logistic_regression(X_train, y_train, X_test, y_test):
    
    parameters = {
        "penalty" : ("l1", "l2"),
        "max_iter" : [(i * 100) for i in range (1, 11)]
    }

    gsc = GridSearchCV(estimator=lr, param_grid=parameters, cv=10, scoring="accuracy") #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    grid_result = gsc.fit(X_train, y_train) #TODO accuracy?
    best_params = grid_result.best_params_
    best_max_iter = best_params["max_iter"]
    best_penalty = best_params["penalty"]
    best_score = grid_result.best_score_

    lr = LogisticRegression(penalty=best_penalty, solver="liblinear", max_iter=best_max_iter) #TODO come back to this
    
    lr_model = lr.fit(X_train, y_train)

    num_test_examples = len(X_test)
    num_test_labels = len(y_test)

    assert num_test_examples == num_test_labels
    
    predictions = []
    
    for test_example in X_test.iterrows(): #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows
        test_example_df = pd.DataFrame(test_example[1]).T #get each row of the df separately
        prediction = lr_model.predict(test_example_df)[0]
        predictions.append(prediction)

    test_model(lr_model, predictions, X_test, y_test)
    
#SVM
def train_and_test_svm(X_train, y_train, X_test, y_test):
    
    parameters = {
        "C" : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "kernel" : ["rbf", "linear", "poly", "sigmoid"],
        "degree" : [1,2,3,4,5],
        "gamma" : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }

    gsc = GridSearchCV(estimator=svm, param_grid=parameters, cv=10, scoring="accuracy") #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    grid_result = gsc.fit(X_train, y_train) #TODO accuracy?
    best_params = grid_result.best_params_
    best_C = best_params["C"]
    best_kernel = best_params["kernel"]
    best_degree = best_params["degree"]
    best_gamma = best_params["gamma"]
    best_score = grid_result.best_score_

    svm = SVC(C=best_C, kernel=best_kernel, degree=best_degree, gamma=best_gamma)
    
    svm_model = svm.fit(X_train, y_train)

    num_test_examples = len(X_test)
    num_test_labels = len(y_test)

    assert num_test_examples == num_test_labels
    
    predictions = []
    
    for test_example in X_test.iterrows(): #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iterrows.html#pandas.DataFrame.iterrows
        test_example_df = pd.DataFrame(test_example[1]).T #get each row of the df separately
        prediction = svm_model.predict(test_example_df)[0]
        predictions.append(prediction)

    test_model(svm_model, predictions, X_test, y_test)


dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test, dataset2_X_train, dataset2_y_train, dataset2_X_test, dataset2_y_test = process_data()

dt = train_and_test_decision_tree(dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test)

train_and_test_boosting(dt, dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test)

train_and_test_logistic_regression(dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test)

train_and_test_svm(dataset1_X_train, dataset1_y_train, dataset1_X_test, dataset1_y_test)
    

