#code for mini batch gradient descent algorithm
import random
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
class MBGDRegressor:
  def __init__(self,batch_size,learning_rate=0.01,epochs=1):
    self.batch_size=batch_size
    self.learning_rate=learning_rate
    self.epochs=epochs
    self.coefficients=None
    self.intercept=None
  def fit(self,x_train,y_train):
    x_train=x_train.values
    y_train=y_train.values
    self.coefficients=np.ones(x_train.shape[1])
    self.intercept=0
    for i in range(self.epochs):
      for j in range(0,int(x_train.shape[0]/self.batch_size)):
        #randomly selecting the batch_size number of rows from the x_train
        random_indices=random.sample(range(x_train.shape[0]),self.batch_size)
        x_batch=x_train[random_indices]
        y_batch=y_train[random_indices]
        #calculating the derivative wrt intercept
        y_pred=np.dot(x_batch,self.coefficients)+self.intercept
        intercept_der=-2*np.sum(y_batch-y_pred)/self.batch_size
        #updating the intercept
        self.intercept-=self.learning_rate*intercept_der
        #calculating the derivative wrt coefficients
        coefficient_der=-2*np.dot(x_batch.T,(y_batch-y_pred))/self.batch_size
        #updating the coefficients
        self.coefficients-=self.learning_rate*coefficient_der
    print(self.coefficients)
    print(self.intercept)
  def predict(self,x_test):
    x_test=x_test.values
    return np.dot(x_test,self.coefficients)+self.intercept
data=pd.read_csv("Student_Performance.csv")
data.drop(data.columns[2],axis=1,inplace=True)
#train test split
x_train,x_test,y_train,y_test=train_test_split(data.iloc[:,0:4],data.iloc[:,-1],random_state=3,test_size=0.2)
mbgd=MBGDRegressor(batch_size=80,learning_rate=0.0001,epochs=61)#epochs=6100
mbgd.fit(x_train,y_train)
print("r2 score",r2_score(y_test,mbgd.predict(x_test)))