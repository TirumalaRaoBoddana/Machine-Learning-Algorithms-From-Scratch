import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
class SGDRegressor:
    def __init__(self, learning_rate=0.0001, epochs=20):
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
            for j in range(n_samples):
                random_index = np.random.randint(0, n_samples)  # Pick a random sample index
                y_pred = np.dot(x_train[random_index], self.coefficients) + self.intercept

                # Calculate gradients
                derivative_intercept = -2 * (y_train[random_index] - y_pred)
                derivative_coefficient = -2 * x_train[random_index] * (y_train[random_index] - y_pred)

                # Update the intercept and coefficients
                self.intercept -= self.learning_rate * derivative_intercept
                self.coefficients -= self.learning_rate * derivative_coefficient

    def predict(self, x_test):
        x_test = x_test.values
        return np.dot(x_test, self.coefficients) + self.intercept
# Loading the dataset
data = pd.read_csv("Student_Performance.csv")
sgd=SGDRegressor(learning_rate=0.0001,epochs=100)
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0:5], data.iloc[:, -1], test_size=0.2, random_state=2)
sgd.fit(x_train.drop(data.columns[2],axis=1),y_train)
print("r2 score=",r2_score(y_test,sgd.predict(x_test.drop(data.columns[2],axis=1))))
print(sgd.coefficients)
print(sgd.intercept)