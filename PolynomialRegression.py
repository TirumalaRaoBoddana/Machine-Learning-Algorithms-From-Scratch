import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
pd.options.display.float_format = '{:.4f}'.format
def get_polynomial_features(features,degree=2):
  feature_indices=[]
  features_df=pd.DataFrame(features)
  for i in range(1,degree+1,1):
    feature_indices.extend(list(itertools.combinations_with_replacement([i for i in range(0,len(features_df.columns))],i)))
  column_names=[]
  columns=[]
  for i in feature_indices:
    #for every combination we need to create a new feature of the product of the features of elements of the tuple ex:(0,0) features_df[0]*features_df[0],(0,1,1) features_df[0]*features_df[1]*features_df[1]
    column_name=""
    column_data=np.ones(features_df.shape[0])
    for j in list(set(i)):
      column_name=column_name+features_df.columns[j]+"^"+str(i.count(j))
      column_data*=(features_df.iloc[:,j]**(i.count(j))) #getting the particular column using the index
    column_names.append(column_name)
    columns.append(column_data)
  poly_features_df = pd.DataFrame(np.column_stack(columns), columns=column_names)
  return poly_features_df
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
student=pd.read_csv("Student_Performance.csv")
x,y=get_polynomial_features(student.select_dtypes("int64"),degree=2),student.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
mlr=MultipleLinearRegression()
mlr.fit(x_train,y_train)
print("coefficients:",mlr.coefficients)
print("intercept:",mlr.intercept)
print("r2 score:",r2_score(y_test,mlr.predict(x_test)))
