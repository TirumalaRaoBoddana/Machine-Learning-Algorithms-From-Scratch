import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
X, y = make_classification(n_samples=100,
                           n_features=2,
                           n_informative=1,
                           n_redundant=0,
                           n_clusters_per_class=1,
                           n_classes=2,
                           random_state=41,class_sep=0.8)
def perceptron(X, y, epochs, learning_rate):
    X = np.insert(X, 0, 1, axis=1)  # Insert bias term (1) into the features
    coefficients = np.ones(X.shape[1])  # Initialize coefficients
    weights_history = []

    for i in range(epochs):
        random_index = np.random.randint(0, X.shape[0])
        y_hat = np.dot(X[random_index], coefficients)
        predicted_value = 1 if y_hat >= 0 else 0 #step function
        coefficients = coefficients + learning_rate * (y[random_index] - predicted_value) * X[random_index]
        weights_history.append(coefficients.copy())  # Save a copy of the coefficients at each step

    return weights_history

# Run Perceptron and get weights over time
epochs = 1000
learning_rate = 0.01
weights_history = perceptron(X, y, epochs, learning_rate)

fig, ax = plt.subplots(figsize=(8, 6))

# Scatter plot for the data points
ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
# Line for the decision boundary (initially empty)
line, = ax.plot([], [], color='green')
# Set labels and title
ax.set_title("Perceptron Decision Boundary Movement")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

# Initialize the plot
def init():
    line.set_data([], [])
    return line,

# Function to update the decision boundary at each frame (epoch)
def update(epoch):
    weights = weights_history[epoch]
    # Slope and intercept of the decision boundary
    slope = -weights[1] / weights[2]
    intercept = -weights[0] / weights[2]
    # Generate x values and calculate corresponding y values
    x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_values = slope * x_values + intercept
    # Update the line
    line.set_data(x_values, y_values)
    return line,
# Create the animation with blit=True and init_func

animation = FuncAnimation(fig, update, frames=len(weights_history), init_func=init, interval=50, blit=True)
# Save as a GIF
plt.show()
