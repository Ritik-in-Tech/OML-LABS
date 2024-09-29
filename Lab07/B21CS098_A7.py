#!/usr/bin/env python
# coding: utf-8

# ### Import libraries
# 

# In[25]:


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


# ### Question1

# In[26]:


def f(x):
    return (1 - x[0])**2 + (x[1] - x[0]**2)**2

def grad_f(x):
    df_dx1 = -2 * (1 - x[0]) - 4 * x[0] * (x[1] - x[0]**2)
    df_dx2 = 2 * (x[1] - x[0]**2)
    return np.array([df_dx1, df_dx2])

def hessian_f(x):
    h11 = 2 - 4 * (x[1] - 3 * x[0]**2)
    h12 = -4 * x[0]
    h21 = h12
    h22 = 2
    return np.array([[h11, h12], [h21, h22]])

def calculate_grad_hess(x, L):
    return grad_f(x), hessian_f(x)

def gradient(L, x):
    return grad_f(x)

def newton_method(L, A, max_iterations=2000):
    x0 = [0, 3]

    itr = 0
    while not (np.linalg.norm(gradient(L, x0)) < 1e-4 or itr >= max_iterations):
        grad, hess = calculate_grad_hess(x0, L)
        try:
            d = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            hess = hess + np.eye(2) * 1e-4
            d = -np.linalg.solve(hess, grad)

        x0 = x0 + d

        print(f'for iteration: {itr+1}')
        print(f'x0 value: {x0}')
        print(f'function evaluation: {L(x0)}')
        print(f'gradient evaluation: {gradient(L, x0)}')
        print()

        itr += 1

    return x0, L(x0)


A = np.array([[1, 0], [0, 1]])

solution, final_value = newton_method(f, A)

print("Solution:", solution)
print("Function value at solution:", final_value)


# ### Question02
# 

# In[27]:


from numpy.linalg import LinAlgError

data =pd.read_csv('../Lab06/diabetes2.csv') # Load the dataset

# print(data.head)

X = data.drop(columns='Outcome').values   # Extract features and labels
y = data['Outcome'].values  # Labels
y = 2 * y - 1  # Convert to {-1, 1}

X = np.hstack([np.ones((X.shape[0], 1)), X]) # Add an intercept term to X

# Sigmoid function
def sigmoid(z):
    return expit(z)

# Logistic loss function
def logistic_loss(x, X, y):
    z = X @ x
    return -np.sum(np.log(sigmoid(y * z)))

# Gradient of the logistic loss
def grad_logistic_loss(x, X, y):
    z = X @ x
    grad = -X.T @ (y * (1 - sigmoid(y * z)))
    return grad

# Hessian of the logistic loss
def hessian_logistic_loss(x, X, y):
    z = X @ x
    S = sigmoid(y * z) * (1 - sigmoid(y * z))
    H = X.T @ np.diag(S) @ X
    return H

# Modified Newton's method with logistic regression
def modified_newton_logistic(X, y, tol=1e-4, max_iter=2000):
    x = np.zeros(X.shape[1])  # Initialize weights
    for i in range(max_iter):
        grad = grad_logistic_loss(x, X, y)
        hess = hessian_logistic_loss(x, X, y)

        # Check if the gradient norm is small enough (stopping criterion)
        if np.linalg.norm(grad) < tol:
            print(f'Converged in {i} iterations.')
            return x
        
        # Check if the Hessian is positive definite
        try:
            np.linalg.cholesky(hess)
        except LinAlgError:
            print("Hessian matrix is not positive definite.")
            return None
        
        # Update the parameter vector x using Newton's method
        delta_x = np.linalg.solve(hess, grad)
        x = x - delta_x

    print("Maximum iterations reached.")
    return x

# Apply the modified Newton method
solution = modified_newton_logistic(X, y)

if solution is not None:
    print(f"Optimized weights: {solution}")
else:
    print("Optimization failed.")


# ### Question03
# 

# In[28]:


def purchase_frequency(x, theta):
    theta1, theta2, theta3 = theta
    return np.exp(theta1 * x) * (np.cos(theta2 * x) + np.sin(theta3 * x))

def mse_loss(X, y, theta, lambda_reg=0.01):
    y_pred = purchase_frequency(X, theta)
    loss = np.mean((y - y_pred) ** 2)
    reg_term = lambda_reg * np.sum(theta**2)
    return loss + reg_term

def gradient(X, y, theta):
    theta1, theta2, theta3 = theta
    y_pred = purchase_frequency(X, theta)
    residual = y_pred - y
    grad_theta1 = np.mean(2 * residual * y_pred * X)
    grad_theta2 = np.mean(2 * residual * (-y_pred * np.sin(theta2 * X) * X))
    grad_theta3 = np.mean(2 * residual * (y_pred * np.cos(theta3 * X) * X))
    return np.array([grad_theta1, grad_theta2, grad_theta3])

def hessian(X, theta):
    theta1, theta2, theta3 = theta
    y_pred = purchase_frequency(X, theta)
    H = np.zeros((3, 3))
    H[0, 0] = np.mean(2 * (X ** 2) * y_pred**2)
    H[1, 1] = np.mean(2 * (X ** 2) * y_pred**2 * np.sin(theta2 * X)**2)
    H[2, 2] = np.mean(2 * (X ** 2) * y_pred**2 * np.cos(theta3 * X)**2)
    return H

def newton_method(X, y, tol=1e-4, max_iter=100):
    theta = np.array([0.5, 0.5, 0.5])
    for i in range(max_iter):
        grad = gradient(X, y, theta)
        H = hessian(X, theta)
        try:
            delta_theta = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            H = H + np.eye(3) * 1e-4
            delta_theta = np.linalg.solve(H, -grad)
        theta = theta + delta_theta
        if np.linalg.norm(grad) < tol:
            print(f"Converged in {i+1} iterations.")
            break
    return theta

df = pd.read_csv('Customer Purchasing Behaviors.csv')
X = df['loyalty_score'].values
y = df['purchase_frequency'].values

optimal_theta = newton_method(X, y)

print("Optimal theta values:", optimal_theta)

R = 98 # last 2 digits of roll no.
x_test = R / 10 + 1
y_estimate = purchase_frequency(x_test, optimal_theta)

print(f"Estimated purchase frequency for x = {x_test}: {y_estimate}")


# ### Question04
# 

# In[29]:


def logistic_function(x1, x2, beta1, beta2):
    z = beta1 * x1 + beta2 * x2
    z_clipped = np.clip(z, -500, 500)
    return np.exp(z_clipped) / (1 + np.exp(z_clipped))

def total_error(beta, X1, X2, y):
    beta1, beta2 = beta
    predictions = logistic_function(X1, X2, beta1, beta2)
    return np.sum((predictions - y) ** 2)

def gradient(beta, X1, X2, y):
    beta1, beta2 = beta
    predictions = logistic_function(X1, X2, beta1, beta2)

    grad_beta1 = 2 * np.sum((predictions - y) * predictions * (1 - predictions) * X1)
    grad_beta2 = 2 * np.sum((predictions - y) * predictions * (1 - predictions) * X2)

    return np.array([grad_beta1, grad_beta2])

def hessian(beta, X1, X2, y):
    beta1, beta2 = beta
    predictions = logistic_function(X1, X2, beta1, beta2)

    hess_beta1_beta1 = 2 * np.sum(predictions * (1 - predictions) * (1 - 2 * predictions) * X1**2)
    hess_beta2_beta2 = 2 * np.sum(predictions * (1 - predictions) * (1 - 2 * predictions) * X2**2)
    hess_beta1_beta2 = 2 * np.sum(predictions * (1 - predictions) * (1 - 2 * predictions) * X1 * X2)

    return np.array([[hess_beta1_beta1, hess_beta1_beta2], [hess_beta1_beta2, hess_beta2_beta2]])

def modified_newton_method(X1, X2, y, beta_init, max_iters=1000, tol=1e-4):
    converged = False
    iteration = 0
    beta = beta_init.copy()
    while not converged and iteration < max_iters:
        grad = gradient(beta, X1, X2, y)
        hess = hessian(beta, X1, X2, y)

        hess += np.eye(2) * 1e-4

        delta_beta = np.linalg.solve(hess, -grad)
        beta += delta_beta

        if np.linalg.norm(delta_beta) < tol:
            converged = True
            print(f"Converged after {iteration + 1} iterations.")

        iteration += 1

    if not converged:
        print(f"Did not converge within {max_iters} iterations.")

    return beta

data = pd.read_csv('new data.csv')

Y = data['Y'].values
X1 = data['x1'].values
X2 = data['x2'].values

beta_init = np.array([0.0, 0.0])

beta_optimal = modified_newton_method(X1, X2, Y, beta_init)
print(f"Optimal beta values: {beta_optimal}")

