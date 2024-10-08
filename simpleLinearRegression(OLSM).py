import numpy as np
import pandas as pd
class SimpleLinearRegression:
  def __init__(self):
    self.m=None
    self.b=None
  def fit(self,x_train,y_train):
    x_train=x_train.values
    y_train=y_train.values
    x_mean=np.mean(x_train)
    y_mean=np.mean(y_train)
    num=0
    den=0
    for i in range(len(x_train)):
      num+=(x_train[i]-x_mean)*(y_train[i]-y_mean)
      den+=(x_train[i]-x_mean)**2
    self.m=num/den
    self.b=y_mean-((self.m)*x_mean)
  def predict(self,x_test):
    return self.m*x_test+self.b