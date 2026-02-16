import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# data collaection and processing


# loading a dataset to a pandas DataFrame
sonar_data = pd.read_csv('sonar_data.csv', header=None)

sonar_data.head()
#print(sonar_data.head())


#number of rows and columns
# print(sonar_data.shape)

#generate descriptive statistics
# print(sonar_data.describe())


#get count of mines and rock data
# print(sonar_data[60].value_counts())


# get mean of every column
# print(sonar_data.groupby(60).mean())



# saperating data and lable
x = sonar_data.drop(columns=60) # stored data without lables
# print(x)
y = sonar_data[60]                      # stored lables only
# print(y)


# training and test data spilit
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.1, stratify=y, random_state=1)
# print(x.shape, x_train.shape, x_test.shape)



