import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_roc
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


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

    prfs = precision_recall_fscore_support(y_test, predictions)
    
    precision = prfs[0]
    recall = prfs[1]
    fscore = prfs[2]

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    proba = model.predict_proba(X_test)
    plot_roc(y_test, proba)
    plt.show()

    return precision, recall, fscore, auc

def kfold_knn (X, y, range_tune=[1,10],fold=10):
    #create new a knn model
    assert range_tune [0] < range_tune[1]
    knn = KNeighborsClassifier()
    #create a dictionary of all values we want to test for n_neighbors
    param_grid = {'n_neighbors': np.arange(range_tune[0], range_tune[1])}
    #use gridsearch to test all values for n_neighbors
    knn_gsf = GridSearchCV(knn, param_grid, cv=fold, refit=True)
    #fit model to data
    knn_gsf.fit(X, y)
    return knn_gsf.best_estimator_, knn_gsf.best_params_
    
def kfold_random_forest(X,y,trees = [1,10], depth=[1,10], fold=10):
    assert trees[0] < trees[1]
    assert depth[0] < depth[1]
    rf = RandomForestClassifier()
    param_grid = {'n_estimators': np.arange(trees[0], trees[1]), 'max_depth': np.arange(depth[0],depth[1])}
    rf_gsf = GridSearchCV(rf, param_grid, cv=fold, refit=True)

    rf_gsf.fit(X,y)
    return rf_gsf.best_estimator_, rf_gsf.best_params_