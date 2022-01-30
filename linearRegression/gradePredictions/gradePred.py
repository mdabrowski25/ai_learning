import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

# LINEAR REGRESSION - model for data with strong correlation
# This program gets the students grades, study time, failures and absences from csv file
# and predicts 3rd student grade based on previous grades and all other students grade pattern

# getting data from csv by pandas
data = pd.read_csv("student-mat.csv", sep=";")

# array with arrays of integers defining student data with correlation
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# defining what we want to predict
predict = "G3"

# arg two which is "1" in data.drop is column deletion
# so X variable is the same as data variable but without 3rd grade
X = np.array(data.drop([predict], 1))

# y array is one dimension array with all 3rd grades of students
y = np.array(data[predict])

# splitting X and y into samples - X stands for data without 3rd grade, y - for only 3rd grade
# we test the correlation between these two
#
# x_train and y_train are 90% of data (defined by test_size = 0.1 variable)
# ---> on that data we draw the linear regression line and on that base we will get predictions
# x_test and y_test are 10% of data (defined by test_size = 0.1 variable)
# ---> on that data we will test accuracy of our model
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# Defining object for linear regression model
linear = linear_model.LinearRegression()

# feeding our model with correlated x and y train data
linear.fit(x_train, y_train)
# getting accuracy percentage (min 0, max 1), we test it on x and y this time with 10% test samples we defined
# x_test is data without 3rd grade, on that data we base our prediction
# y_test are correct answers
# our method will check the percentage based on checking if calculated answer (based on x_test) == y_test answer
acc = linear.score(x_test, y_test)
print('Accuracy:', acc)
print('\n')

# our data x_train data has 5 columns of data, so we get 5 Coefficient as our model has 5 dimensions
print("Coefficient: \n", linear.coef_)
# slope of our data ---> two dimensional pattern: y = mx + b ---> Intercept stands for m
print("Intercept: \n", linear.intercept_)
print('\n')

# making predictions for x_test data - it contains 5 columns of data, we predict 3rd grade based on that
predictions = linear.predict(x_test)

# printing pattern: prediction [sub-array for student data without 3rd grade] 3rd grade
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
