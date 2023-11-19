import pandas as pd

#Read Data 1
df1 = pd.read_csv("data1.csv")
df1.columns=["x"+str(i) for i in range(len(df1.columns))]

df1.rename(columns = {list(df1)[len(df1.columns)-1]:'label'}, inplace=True)

#Read Data 2
df2 = pd.read_csv("data2.csv")
df2.columns=["x"+str(i) for i in range(len(df2.columns))]

df2.rename(columns = {list(df2)[len(df2.columns)-1]:'label'}, inplace=True)
