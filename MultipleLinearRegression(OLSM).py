import numpy as np
import pandas as pd
class MultipleLinearRegression:
  def __init__(self):
    self.coefficients=None
    self.coef=None
    self.intercept=None
  def fit(self,x_train,y_train):
    x_train=x_train.values
    y_train=y_train.values
    x_train=np.hstack((np.ones((x_train.shape[0],1)),x_train))
    x_train_transpose=x_train.T
    inverse_matirix=np.linalg.inv(np.dot(x_train_transpose,x_train))
    self.coefficients=np.dot(np.dot(inverse_matirix,x_train.T),y_train)
    self.coef=self.coefficients[1:]
    self.intercept=self.coefficients[0]
  def predict(self,x_test):
    x_test=x_test.values
    x_test=np.hstack((np.ones((x_test.shape[0],1)),x_test))
    return np.dot(x_test,self.coefficients)
