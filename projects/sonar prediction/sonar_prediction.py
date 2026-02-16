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



# model training using logistic regression
model = LogisticRegression()

# training regression model with training data
model.fit(x_train, y_train)




# mpdel evaluation


# accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('Accuracy on training data : ', training_data_accuracy)



# accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('Accuracy on test data : ', test_data_accuracy)




# Making prediction system


# giving one input example
input_data = (0.0516,0.0944,0.0622,0.0415,0.0995,0.2431,0.1777,0.2018,0.2611,0.1294,0.2646,0.2778,0.4432,0.3672,0.2035,0.2764,0.3252,0.1536,0.2784,0.3508,0.5187,0.7052,0.7143,0.6814,0.5100,0.5308,0.6131,0.8388,0.9031,0.8607,0.9656,0.9168,0.7132,0.6898,0.7310,0.4134,0.1580,0.1819,0.1381,0.2960,0.6935,0.8246,0.5351,0.4403,0.6448,0.6214,0.3016,0.1379,0.0364,0.0355,0.0456,0.0432,0.0274,0.0152,0.0120,0.0129,0.0020,0.0109,0.0074,0.0078)


# changing input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data) 


# reshape numpy array as we are predicting for one instance
input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshape)

# prediction result
# print(prediction)
if(prediction[0]=='R'):
    print('The object is Rock')
else:
    print('The object is Mine')