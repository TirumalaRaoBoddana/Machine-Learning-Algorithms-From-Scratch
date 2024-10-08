import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
"""Why Use Gradient Descent Over the Least Squares Method for Multiple Linear Regression?
    1.Computational Efficiency with Large Datasets:Least Squares requires calculating the inverse of the matrix  (X^T)X ,where X is the matrix of input features.This operation has a time complexity of O( n3 ) where n is the number of features.
    2.Flexibility with Different Cost Functions:Least Squares is limited to only cost functionn associated with the regression problem.but Gradient descent can minimize any type of the Cost Function
"""
#coding the gradient descent for multiple fetures
class BGDRegressor:
    def __init__(self, learning_rate=0.001, epochs=500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coefficients = None
        self.intercept = None
    def fit(self, x_train, y_train):
        x_train = x_train.values
        y_train = y_train.values
        n_samples, n_features = x_train.shape
        # Initialize coefficients and intercept
        self.intercept = 0
        self.coefficients = np.zeros(n_features)  # Start with zeros instead of ones
        # Gradient Descent Loop
        for i in range(self.epochs):
            y_pred = np.dot(x_train, self.coefficients) + self.intercept  # Predicted values
            # Calculate gradients
            derivative_intercept = (-2 / n_samples) * np.sum(y_train-y_pred)
            derivative_coefficients = (-2 / n_samples) * np.dot(x_train.T, y_train-y_pred)
            # Update intercept and coefficients
            self.intercept -= self.learning_rate * derivative_intercept
            self.coefficients -= self.learning_rate * derivative_coefficients
    def predict(self, x_test):
        x_test = x_test.values
        return np.dot(x_test, self.coefficients) + self.intercept
#loading the dataset
data=pd.read_csv("Student_Performance.csv")
#splitting the data into test data and train data
x_train,x_test,y_train,y_test=train_test_split(data.iloc[0:,0:len(data.columns)-1],data.iloc[:,-1])
gd=BGDRegressor(learning_rate=0.0001,epochs=100)#put epochs to 3,90,000 to get the appoximated intercepts and coefficients
gd.fit(x_train.drop(data.columns[2],axis=1),y_train)
print("coefficients:",gd.coefficients)
print("intercept",gd.intercept)
print("r2 score=",r2_score(y_test,gd.predict(x_test.drop(data.columns[2],axis=1))))