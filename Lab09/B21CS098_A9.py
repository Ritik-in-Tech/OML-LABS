#!/usr/bin/env python
# coding: utf-8

# ### Question 1
# 

# #### L1 Regularization
# 

# In[31]:


import numpy as np
def soft_thresholding(x, alpha):
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

def proximal_gradient_method(initial_x, max_iterations=1000, alpha_type="fixed", r=1, tol=1e-6):
    x = np.array(initial_x, dtype=float)
    history = [x.copy()]
    
    for k in range(max_iterations):
        if alpha_type == "fixed":
            alpha_k = 1/2
        elif alpha_type == "variable":
            alpha_k = 1 / (r + k)
        else:
            raise ValueError("alpha_type must be 'fixed' or 'variable'")
        
        # Gradient of the smooth part
        gradient_g = np.array([x[0] - 2, x[1] - 2])
        
        # Gradient step
        x_gradient_step = x - alpha_k * gradient_g
        
        # Proximal step (using soft-thresholding for L1 norm)
        x_next = soft_thresholding(x_gradient_step, alpha_k * 0.5)
        
        # Check for convergence
        if np.linalg.norm(x_next - x) < tol:
            break
        
        # Update x
        x = x_next
        history.append(x.copy())
    
    return x, history

# Parameters
initial_x = [0.0, 0.0]  # Initial guess
r = 8  # Example last digit of roll number

# Solve with fixed step size
solution_fixed, history_fixed = proximal_gradient_method(initial_x, max_iterations=100, alpha_type="fixed", r=r)

# Solve with variable step size
solution_variable, history_variable = proximal_gradient_method(initial_x, max_iterations=100, alpha_type="variable", r=r)

# Results
print("Solution with fixed step size:", solution_fixed)
print("Solution with variable step size:", solution_variable)


# #### L2 Regularization
# 

# In[32]:


import numpy as np

def l2_proximal_operator(x, alpha):
    """
    Proximal operator for L2 regularization (ridge)
    For L2 norm, the proximal operator is: x / (1 + alpha)
    """
    return x / (1 + alpha)

def proximal_gradient_method_l2(initial_x, max_iterations=1000, alpha_type="fixed", r=1, tol=1e-6):
    x = np.array(initial_x, dtype=float)
    history = [x.copy()]
    
    for k in range(max_iterations):
        if alpha_type == "fixed":
            alpha_k = 1/2
        elif alpha_type == "variable":
            alpha_k = 1 / (r + k)
        else:
            raise ValueError("alpha_type must be 'fixed' or 'variable'")
        
        # Gradient of the smooth part (quadratic loss)
        gradient_g = np.array([x[0] - 2, x[1] - 2])
        
        # Gradient step
        x_gradient_step = x - alpha_k * gradient_g
        
        # Proximal step (using L2 proximal operator)
        # The 0.5 factor is the regularization parameter (similar to your original code)
        x_next = l2_proximal_operator(x_gradient_step, alpha_k * 0.5)
        
        # Check for convergence
        if np.linalg.norm(x_next - x) < tol:
            break
        
        # Update x
        x = x_next
        history.append(x.copy())
    
    return x, history

# Parameters
initial_x = [0.0, 0.0]  # Initial guess
r = 8  # Example last digit of roll number

# Solve with fixed step size
solution_fixed, history_fixed = proximal_gradient_method_l2(initial_x, max_iterations=100, alpha_type="fixed", r=r)

# Solve with variable step size
solution_variable, history_variable = proximal_gradient_method_l2(initial_x, max_iterations=100, alpha_type="variable", r=r)

# Results
print("Solution with fixed step size:", solution_fixed)
print("Solution with variable step size:", solution_variable)


# ### Question 2
# 

# ##### L1 Regularization
# 

# In[33]:


import numpy as np
import pandas as pd

def soft_thresholding(x, alpha):
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

def compute_step_size(A):
    eigvals = np.linalg.eigvalsh(A.T @ A)
    max_eigenvalue = np.max(eigvals)
    # Step size alpha_k
    alpha_k = 1 / (max_eigenvalue / 2 + 0.5)
    return alpha_k

def proximal_gradient_method(A, y, lambda_val, max_iterations=1000, tol=1e-6):
    # Initialize x
    x = np.zeros(A.shape[1])
    history = [x.copy()]

    # Compute the step size
    alpha_k = compute_step_size(A)

    for k in range(max_iterations):
        # Gradient step
        gradient = A.T @ (A @ x - y)
        x_gradient_step = x - alpha_k * gradient

        # Proximal step (soft-thresholding for L1 norm)
        x_next = soft_thresholding(x_gradient_step, alpha_k * lambda_val)

        # Check for convergence
        if np.linalg.norm(x_next - x) < tol:
            break

        # Update x
        x = x_next
        history.append(x.copy())

    return x, history


data=pd.read_csv('./train_a.csv')

A = data[['Value1','Value2']].values
y = data['Result'].values

# Add a column of ones to A for the intercept term
A = np.hstack([A, np.ones((A.shape[0], 1))])


R = 8  # last digit of roll number
lambda_val = abs(R / 10 - 5)

# Solve the optimization problem
solution, history = proximal_gradient_method(A, y, lambda_val)

# Extract coefficients and intercept
a = solution[:2]
beta = solution[2]

# Output the results
print("Coefficients a:", a)
print("Intercept beta:", beta)
print("Solution is:", solution)



# #### L2 Regularization
# 

# In[34]:


import numpy as np
import pandas as pd

def compute_step_size(A):
    eigvals = np.linalg.eigvalsh(A.T @ A)
    max_eigenvalue = np.max(eigvals)
    # Step size alpha_k
    alpha_k = 1 / (max_eigenvalue / 2 + 0.5)
    return alpha_k

def proximal_gradient_method_l2(A, y, lambda_val, max_iterations=1000, tol=1e-6):
    # Initialize x
    x = np.zeros(A.shape[1])
    history = [x.copy()]

    # Compute the step size
    alpha_k = compute_step_size(A)

    for k in range(max_iterations):
        # Gradient of the smooth part (least squares loss + L2 regularization)
        gradient = A.T @ (A @ x - y) + lambda_val * x
        x_next = x - alpha_k * gradient

        # Check for convergence
        if np.linalg.norm(x_next - x) < tol:
            break

        # Update x
        x = x_next
        history.append(x.copy())

    return x, history

# Load and prepare data
data = pd.read_csv('./train_a.csv')

A = data[['Value1','Value2']].values
y = data['Result'].values

# Add a column of ones to A for the intercept term
A = np.hstack([A, np.ones((A.shape[0], 1))])

# Set lambda value based on roll number
R = 8  # last digit of roll number
lambda_val = abs(R / 10 - 5)

# Solve the optimization problem
solution, history = proximal_gradient_method_l2(A, y, lambda_val)

# Extract coefficients and intercept
a = solution[:2]
beta = solution[2]

# Output the results
print("Coefficients a:", a)
print("Intercept beta:", beta)
print("Solution is:", solution)


# ### Question 3
# 

# #### L1 Regularization
# 

# In[35]:


import numpy as np
import pandas as pd

def soft_thresholding(x, alpha):
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

def compute_step_size(A):
    eigvals = np.linalg.eigvalsh(A.T @ A)
    max_eigenvalue = np.max(eigvals)
    alpha_k = 1 / (max_eigenvalue / 2 + 0.5)
    return alpha_k

def proximal_gradient_method(A, y, lambda_val, max_iterations=1000, tol=1e-6):
    # Initialize x
    x = np.zeros(A.shape[1])
    history = [x.copy()]

    # Compute the step size
    alpha_k = compute_step_size(A)

    for k in range(max_iterations):
        # Gradient step
        gradient = A.T @ (A @ x - y)
        x_gradient_step = x - alpha_k * gradient

        # Proximal step (soft-thresholding for L1 norm)
        x_next = soft_thresholding(x_gradient_step, alpha_k * lambda_val)

        # Check for convergence
        if np.linalg.norm(x_next - x) < tol:
            break

        # Update x
        x = x_next
        history.append(x.copy())

    return x, history


data=pd.read_csv('./4 columns.csv')

A = data[['area','bedrooms','bathrooms']].values
y = data['price'].values

# Add a column of ones to A for the intercept term
A = np.hstack([A, np.ones((A.shape[0], 1))])


R = 8  # last digit of roll number
lambda_val = abs(R / 10 - 5)

# Solve the optimization problem
solution, history = proximal_gradient_method(A, y, lambda_val)

# Extract coefficients and intercept
a = solution[:3]
beta = solution[3]

# Output the results
print("Coefficients [area, bedrooms, bathrooms]:", a)
print("Intercept beta:", beta)
print("Solution is:", solution)



# #### L2 Regularization
# 

# In[36]:


import numpy as np
import pandas as pd

def compute_step_size(A):
    eigvals = np.linalg.eigvalsh(A.T @ A)
    max_eigenvalue = np.max(eigvals)
    alpha_k = 1 / (max_eigenvalue / 2 + 0.5)
    return alpha_k

def proximal_gradient_method_l2(A, y, lambda_val, max_iterations=1000, tol=1e-6):
    # Initialize x
    x = np.zeros(A.shape[1])
    history = [x.copy()]

    # Compute the step size
    alpha_k = compute_step_size(A)

    for k in range(max_iterations):
        # Gradient of the loss function (MSE + L2 regularization)
        gradient = A.T @ (A @ x - y) + lambda_val * x
        
        
        x_next = x - alpha_k * gradient

        # Check for convergence
        if np.linalg.norm(x_next - x) < tol:
            break

        # Update x
        x = x_next
        history.append(x.copy())

    return x, history

# Load and prepare data
data = pd.read_csv('./4 columns.csv')

# Prepare feature matrix A and target vector y
A = data[['area', 'bedrooms', 'bathrooms']].values
y = data['price'].values


A = np.hstack([A, np.ones((A.shape[0], 1))])

R = 8   
lambda_val = abs(R / 10 - 5)

solution, history = proximal_gradient_method_l2(A, y, lambda_val)

a = solution[:3]  
beta = solution[3]   

# Output the results
print("Coefficients [area, bedrooms, bathrooms]:", a)
print("Intercept:", beta)
print("Complete solution:", solution)


# ### Question 4
# 

# #### L1 Regularization
# 

# In[37]:


import numpy as np
import pandas as pd
from scipy.optimize import minimize

data = pd.read_csv('./Customer Purchasing Behaviors (1).csv')

x = data['loyalty_score'].values
y = data['purchase_frequency'].values

R = 98
target_x = R / 10 + 1

lambda_value = abs(R / 10 - 5)

def model(theta, x):
    return np.exp(theta[0] * x) * (np.cos(theta[1] * x) + np.sin(theta[2] * x))

def objective(theta):
    residual = model(theta, x) - y
    return 0.5 * np.sum(residual**2) + (lambda_value / 2) * np.sum(np.abs(theta))

theta0 = np.array([0.1, 0.1, 0.1])

result = minimize(objective, theta0, method='L-BFGS-B')

optimal_theta = result.x
print("Optimal θ values:")
print(f"θ1: {optimal_theta[0]:.2f}")
print(f"θ2: {optimal_theta[1]:.2f}")
print(f"θ3: {optimal_theta[2]:.2f}")
print(f"λ: {lambda_value:.2f}")

predicted_purchase_frequency = model(optimal_theta, target_x)
print(f"Estimated purchase frequency for loyalty score {target_x:.2f}: {predicted_purchase_frequency:.2f}")


# #### L2 Regularization
# 

# In[38]:


import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('./Customer Purchasing Behaviors (1).csv')

x = data['loyalty_score'].values
y = data['purchase_frequency'].values

# Parameters based on roll number
R = 98
target_x = R / 10 + 1
lambda_value = abs(R / 10 - 5)

def model(theta, x):
    return np.exp(theta[0] * x) * (np.cos(theta[1] * x) + np.sin(theta[2] * x))

def objective(theta):
    residual = model(theta, x) - y
    l2_penalty = (lambda_value / 2) * np.sum(theta**2)
    return 0.5 * np.sum(residual**2) + l2_penalty

# Initial guess for parameters
theta0 = np.array([0.1, 0.1, 0.1])

# Optimize using L-BFGS-B method
result = minimize(objective, theta0, method='L-BFGS-B')

# Extract optimal parameters
optimal_theta = result.x

# Print results
print("\nOptimization Results:")
print("Optimal θ values:")
print(f"θ1: {optimal_theta[0]:.4f}")
print(f"θ2: {optimal_theta[1]:.4f}")
print(f"θ3: {optimal_theta[2]:.4f}")
print(f"λ: {lambda_value:.4f}")


predicted_purchase_frequency = model(optimal_theta, target_x)
print(f"Estimated purchase frequency for loyality score {target_x}: {predicted_purchase_frequency:.4f}")

