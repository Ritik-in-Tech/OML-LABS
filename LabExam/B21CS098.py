#!/usr/bin/env python
# coding: utf-8

# ## Question 1
# 

# In[1]:


import numpy as np
def f(x):
    x1, x2, x3 = x
    return 4 * x1**2 + 3 * x2**2 + 5 * x3**2 - 3 * x1 * x2 + 4 * x2 * x3 - x1 * x3 + 4 * x1 - 5 * x2


def grad_f(x):
    x1, x2, x3 = x
    df_dx1 = 8 * x1 - 3 * x2 - x3 + 4
    df_dx2 = -3 * x1 + 6 * x2 + 4 * x3 - 5
    df_dx3 = 10 * x3 + 4 * x2 - x1
    return np.array([df_dx1, df_dx2, df_dx3])


r = 8  # Last digit of my roll number (B21CS098)
x0 = np.array([2 + r / 10, 3 - r / 10, 6])  # Initial point as given in the question
tol = 1e-4  # Tolerance for stopping criteria ( given in the question)
max_iter = 1000  # Maximum number of iterations (I am assuming this)
alpha = 1e-1  # Initial step size for line search (I have assumed this)


x = x0
for i in range(max_iter):
    gradient = grad_f(x)
    grad_norm = np.linalg.norm(gradient)
    if grad_norm < tol:
        print(f"Converged in {i} iterations.")
        break
    t = alpha
    beta = 0.5  
    c = 1e-4  
    while f(x - t * gradient) > f(x) - c * t * grad_norm**2:
        t *= beta
    x = x - t * gradient


print("Optimal point:", x)
print("Optimal function value:", f(x))
print("Gradient norm at optimal point:", grad_norm)


# ## Question 2
# 

# In[2]:


import numpy as np
import pandas as pd
data = pd.read_csv("multiple_linear_regression_dataset - Copy (1).csv")
X = data[['age', 'experience']].values
y = data['Income'].values
R = 98  # R=B21CS098 ( As 98 is my last digit of the rollNumber)
lambda_reg = abs(R) / 100 - 0.5  # Given in the question 
alpha = np.array([1.0, 1.0]) # Taken the initial point 
tol = 1e-4 # Tolerance factor
max_iter = 1000  # Max number of iterations
learning_rate = 0.001  # learning rate I have assumed this

def model(x, alpha):
    return alpha[0] * x[:, 0] + np.exp(np.clip(alpha[1] * x[:, 1], -50, 50))

def objective(alpha, X, y, lambda_reg):
    predictions = model(X, alpha)
    mse = np.mean((y - predictions) ** 2)
    l1_norm = lambda_reg * np.sum(np.abs(alpha))
    return mse + l1_norm

def gradient(alpha, X, y):
    predictions = model(X, alpha)
    error = predictions - y
    grad_alpha1 = np.mean(2 * error * X[:, 0])
    grad_alpha2 = np.mean(2 * error * np.exp(np.clip(alpha[1] * X[:, 1], -50, 50)) * X[:, 1])
    grad_alpha1 = np.clip(grad_alpha1, -1e6, 1e6)
    grad_alpha2 = np.clip(grad_alpha2, -1e6, 1e6)
    return np.array([grad_alpha1, grad_alpha2])


def proximal_operator(alpha, learning_rate, lambda_reg):
    return np.sign(alpha) * np.maximum(0, np.abs(alpha) - learning_rate * lambda_reg)

for i in range(max_iter):
    grad = gradient(alpha, X, y)
    alpha = alpha - learning_rate * grad  
    alpha = proximal_operator(alpha, learning_rate, lambda_reg)  
    if np.linalg.norm(grad) < tol:
        print(f"Converged in {i} iterations.")
        break

print("Optimal alpha is:", alpha)
print("Objective function value is:", objective(alpha, X, y, lambda_reg))

