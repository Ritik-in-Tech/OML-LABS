#!/usr/bin/env python
# coding: utf-8

# ### Question01
# 

# In[7]:


import numpy as np
def f(x): # Objective function
    x1, x2 = x
    return (1 - x1)**2 + (x2 - x1**2)**2

def grad_f(x): # Gradient of the objective function
    x1, x2 = x
    df_dx1 = -2 * (1 - x1) - 4 * x1 * (x2 - x1**2)
    df_dx2 = 2 * (x2 - x1**2)
    return np.array([df_dx1, df_dx2])

def hessian_f(x): # Hessian of the objective function
    x1, x2 = x
    d2f_dx1x1 = 2 - 4 * (x2 - x1**2) + 8 * x1**2
    d2f_dx1x2 = -4 * x1
    d2f_dx2x2 = 2
    return np.array([[d2f_dx1x1, d2f_dx1x2], [d2f_dx1x2, d2f_dx2x2]])

def backtracking_line_search(f, grad_f, x, p, alpha=0.3, beta=0.8): # Inexact line search using backtracking
    t = 1
    while f(x + t * p) > f(x) + alpha * t * np.dot(grad_f(x), p):
        t *= beta
    return t

def modified_newton_method(f, grad_f, hessian_f, x0, tol=1e-4, max_iter=2000): # Modified Newton's method
    x = np.array(x0)
    for i in range(max_iter):
        grad = grad_f(x)
        hess = hessian_f(x)

        # Check the stopping criterion
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f'Converged in {i} iterations.')
            return x, f(x)

        # Check if Hessian is positive definite using Cholesky decomposition
        try:
            np.linalg.cholesky(hess)
        except np.linalg.LinAlgError:
            return "Hessian matrix is not positive definite"
        
        # Compute the Newton direction
        p = -np.linalg.solve(hess, grad)
        
        # Perform backtracking line search
        t = backtracking_line_search(f, grad_f, x, p)
        
        # Update the solution
        x = x + t * p
    
    print('Maximum iterations reached.')
    return x, f(x)

x0 = [0, 3] # Initial approximation

 
result = modified_newton_method(f, grad_f, hessian_f, x0) # Run the modified Newton method

if isinstance(result, str):
    print(result)
else:
    solution, f_value = result
    print(f"Solution: {solution}")
    print(f"Function value at the solution: {f_value}")


# ### Question02
# 

# In[8]:


import numpy as np
import pandas as pd
from scipy.special import expit   
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

# In[9]:


import numpy as np
from numpy.linalg import LinAlgError

# Sample dataset
data = pd.read_csv('./Customer Purchasing Behaviors.csv')

# Extract loyalty score (x) and purchase frequency (y)
x = data['loyalty_score'].values
y = data['purchase_frequency'].values

# Define the model for purchase frequency
def model(theta, x):
    theta1, theta2, theta3 = theta
    return np.exp(theta1 * x) * (np.cos(theta2 * x) + np.sin(theta3 * x))

# Loss function (squared error)
def loss_function(theta, x, y):
    y_pred = model(theta, x)
    return np.sum((y - y_pred) ** 2)

# Gradient of the loss function
def grad_loss_function(theta, x, y):
    theta1, theta2, theta3 = theta
    y_pred = model(theta, x)
    error = y - y_pred
    
    # Partial derivatives
    dtheta1 = -2 * np.sum(error * (x * np.exp(theta1 * x)) * (np.cos(theta2 * x) + np.sin(theta3 * x)))
    dtheta2 = -2 * np.sum(error * np.exp(theta1 * x) * (-x * np.sin(theta2 * x)))
    dtheta3 = -2 * np.sum(error * np.exp(theta1 * x) * (x * np.cos(theta3 * x)))
    
    return np.array([dtheta1, dtheta2, dtheta3])

# Hessian of the loss function
def hessian_loss_function(theta, x, y):
    theta1, theta2, theta3 = theta
    y_pred = model(theta, x)
    error = y - y_pred
    
    # Second order partial derivatives
    d2theta1 = 2 * np.sum((x ** 2) * np.exp(2 * theta1 * x) * (np.cos(theta2 * x) + np.sin(theta3 * x)) ** 2)
    d2theta2 = 2 * np.sum(error * np.exp(theta1 * x) * (x ** 2) * np.cos(theta2 * x))
    d2theta3 = 2 * np.sum(error * np.exp(theta1 * x) * (x ** 2) * -np.sin(theta3 * x))
    
    # Cross partial derivatives
    dtheta1theta2 = 2 * np.sum(error * x * np.exp(theta1 * x) * (-x * np.sin(theta2 * x)))
    dtheta1theta3 = 2 * np.sum(error * x * np.exp(theta1 * x) * (x * np.cos(theta3 * x)))
    dtheta2theta3 = 2 * np.sum(error * np.exp(theta1 * x) * (-x ** 2 * np.sin(theta2 * x) * np.cos(theta3 * x)))
    
    # Hessian matrix
    hess = np.array([
        [d2theta1, dtheta1theta2, dtheta1theta3],
        [dtheta1theta2, d2theta2, dtheta2theta3],
        [dtheta1theta3, dtheta2theta3, d2theta3]
    ])
    
    return hess

# Modified Newton's method
def modified_newton(x, y, tol=1e-4, max_iter=100):
    theta = np.zeros(3)  # Initialize theta1, theta2, theta3
    
    for i in range(max_iter):
        grad = grad_loss_function(theta, x, y)
        hess = hessian_loss_function(theta, x, y)
        
        # Check if the gradient norm is small enough (stopping criterion)
        if np.linalg.norm(grad) < tol:
            print(f'Converged in {i} iterations.')
            return theta
        
        # Check if the Hessian is positive definite
        try:
            np.linalg.cholesky(hess)
        except LinAlgError:
            print("Hessian matrix is not positive definite.")
            return None
        
        # Update the parameters using Newton's method
        delta_theta = np.linalg.solve(hess, grad)
        theta = theta - delta_theta
    
    print("Maximum iterations reached.")
    return theta

# Apply the modified Newton method
theta_optimal = modified_newton(x, y)

# If optimization succeeded, estimate purchase frequency for R/10 + 1 where R = 98
if theta_optimal is not None:
    R = 98
    x_new = R / 10 + 1
    purchase_frequency_estimate = model(theta_optimal, x_new)
    print(f"Estimated purchase frequency for loyalty score {x_new}: {purchase_frequency_estimate}")
else:
    print("Optimization failed.")


# ### Question04
# 

# In[10]:


import numpy as np
import pandas as pd

def load_data_from_csv(file_path): # Load the CSV file using pandas
    data = pd.read_csv(file_path)
    x1 = data['x1'].values
    x2 = data['x2'].values
    y = data['Y'].values
    return x1, x2, y

def model(beta, x1, x2): # Define the model for prediction
    beta1, beta2 = beta
    exp_term = np.exp(beta1 * x1 + beta2 * x2)
    return exp_term / (1 + exp_term)


def loss_function(beta, x1, x2, y): # Loss function (squared error)
    y_pred = model(beta, x1, x2)
    return np.sum((y - y_pred) ** 2)

 
def grad_loss_function(beta, x1, x2, y): # Gradient of the loss function
    beta1, beta2 = beta
    y_pred = model(beta, x1, x2)
    error = y - y_pred
    
    # Partial derivatives
    d_beta1 = -2 * np.sum(error * (x1 * y_pred * (1 - y_pred)))
    d_beta2 = -2 * np.sum(error * (x2 * y_pred * (1 - y_pred)))
    
    return np.array([d_beta1, d_beta2])

# Hessian of the loss function
def hessian_loss_function(beta, x1, x2, y):
    beta1, beta2 = beta
    y_pred = model(beta, x1, x2)
    error = y - y_pred
    
    # Second order partial derivatives
    d2_beta1 = 2 * np.sum((x1 ** 2) * y_pred * (1 - y_pred) * (1 - 2 * y_pred))
    d2_beta2 = 2 * np.sum((x2 ** 2) * y_pred * (1 - y_pred) * (1 - 2 * y_pred))
    
    # Cross partial derivatives
    d_beta1_beta2 = 2 * np.sum(x1 * x2 * y_pred * (1 - y_pred) * (1 - 2 * y_pred))
    
    # Hessian matrix
    hess = np.array([
        [d2_beta1, d_beta1_beta2],
        [d_beta1_beta2, d2_beta2]
    ])
    
    return hess

# Modified Newton's method
def modified_newton(x1, x2, y, tol=1e-4, max_iter=100):
    beta = np.zeros(2)  # Initialize beta1 and beta2
    
    for i in range(max_iter):
        grad = grad_loss_function(beta, x1, x2, y)
        hess = hessian_loss_function(beta, x1, x2, y)
        
        # Check if the gradient norm is small enough (stopping criterion)
        if np.linalg.norm(grad) < tol:
            print(f'Converged in {i} iterations.')
            return beta
        
        try:
            delta_beta = np.linalg.solve(hess, grad)
            beta = beta - delta_beta
        except:
            print(f"Singular matrix encountered at iteration {i}.")
            return None
    
    print("Maximum iterations reached.")
    return beta

# Main function to load data, run optimization, and estimate beta values
def main(csv_file_path):
    # Load data from the given CSV file
    x1, x2, y = load_data_from_csv(csv_file_path)
    
    # Apply the modified Newton method
    beta_optimal = modified_newton(x1, x2, y)
    
    if beta_optimal is not None:
        print(f"Optimal beta values: {beta_optimal}")
    else:
        print("Optimization failed.")

# Run the main function with the path to your CSV file
csv_file_path = './new data.csv'  # Replace with the actual file path
main(csv_file_path)

