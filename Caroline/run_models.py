#All models are from sklearn:
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html and 
# SVM link here

#Make train and test split for each dataset
#Write a function to graph and record (and store in a file) accuracy as the model is training AND as the model is testing
#Run the model on multiple different amounts of dtaa for the above and with different hyperparameters

import pandas as pd

#Read data from csv

#Read Data 1
df1 = pd.read_csv("../data1.csv")
df1.columns=["x"+str(i) for i in range(len(df1.columns))]
df1.rename(columns = {list(df1)[len(df1.columns)-1]:'label'}, inplace=True)

#Read Data 2
df2 = pd.read_csv("../data2.csv")
df2.columns=["x"+str(i) for i in range(len(df2.columns))]
df2.rename(columns = {list(df2)[len(df2.columns)-1]:'label'}, inplace=True)

