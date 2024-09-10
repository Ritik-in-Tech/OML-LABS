#!/usr/bin/env python
# coding: utf-8

# ### Import code
# 

# In[20]:


import numpy as np


# ### Question 1
# ![image.png](attachment:image.png)

# In[21]:


def f(x, r1):
    return (x[0] - r1)**4 + (x[0] - 2*x[1])**2

def gradient_f(x, r1):
    dx1 = 4*(x[0] - r1)**3 + 2*(x[0] - 2*x[1])
    dx2 = -4*(x[0] - 2*x[1])
    return np.array([dx1, dx2])

def armijo_wolfe_line_search(f, grad_f, x, d, r1, alpha=1.0, c1=1e-4, c2=0.9, max_iter=25):
    phi = lambda alpha: f(x + alpha * d, r1)
    dphi = lambda alpha: np.dot(grad_f(x + alpha * d, r1), d)
    
    phi_0 = phi(0)
    dphi_0 = dphi(0)
    
    for _ in range(max_iter):
        if phi(alpha) <= phi_0 + c1 * alpha * dphi_0 and dphi(alpha) >= c2 * dphi_0:
            return alpha
        alpha *= 0.5
    
    return alpha

def descent_method(x0, r1, beta1, r, epsilon, max_iter):
    x = x0
    iter_count = 0
    func_evals = 0
    grad_evals = 0
    
    while iter_count < max_iter:
        grad = gradient_f(x, r1)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < epsilon:
            break
        
        d = -grad
        alpha = armijo_wolfe_line_search(f, gradient_f, x, d, r1)
        
        x = x + alpha * d
        beta1 *= r
        
        iter_count += 1
        func_evals += 1   
        grad_evals += 1   
    
    return x, iter_count, func_evals, grad_evals

# Parameters
r1 = 8   
x0 = np.array([0.5, 0.25])
beta1 = 1e-4
r = 0.5
epsilon = 10**(-3.5)
max_iter = 500

# Run the descent method
result, iterations, func_evals, grad_evals = descent_method(x0, r1, beta1, r, epsilon, max_iter)

print(f"Number of iterations: {iterations}")
print(f"Number of function evaluations: {func_evals}")
print(f"Number of gradient evaluations: {grad_evals}")
print(f"Final solution: {result}")
print(f"Final function value: {f(result, r1)}")
print(f"Final gradient norm: {np.linalg.norm(gradient_f(result, r1))}")


# ### Question 2
# ![image.png](attachment:image.png)

# In[22]:


def f(x, r1):
    return (x[0] - r1)**4 + (x[0] - 2*x[1])**2

def gradient_f(x, r1):
    dx1 = 4*(x[0] - r1)**3 + 2*(x[0] - 2*x[1])
    dx2 = -4*(x[0] - 2*x[1])
    return np.array([dx1, dx2])

def armijo_wolfe_line_search(f, grad_f, x, d, r1, alpha=1.0, c1=1e-4, c2=0.9, max_iter=25):
    phi = lambda alpha: f(x + alpha * d, r1)
    dphi = lambda alpha: np.dot(grad_f(x + alpha * d, r1), d)
    
    phi_0 = phi(0)
    dphi_0 = dphi(0)
    
    for _ in range(max_iter):
        if phi(alpha) <= phi_0 + c1 * alpha * dphi_0 and dphi(alpha) >= c2 * dphi_0:
            return alpha
        alpha *= 0.5
    
    return alpha

def descent_method(x0, r1, beta1, r, epsilon, max_iter, method):
    x = x0
    iter_count = 0
    func_evals = 0
    grad_evals = 0
    
    B = np.array([[2*r1, np.sqrt(r1)], [np.sqrt(r1), r1]])
    B_inv = np.linalg.inv(B)
    
    while iter_count < max_iter:
        grad = gradient_f(x, r1)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < epsilon:
            break
        
        if method == 'B_inv':
            d = -np.dot(B_inv, grad)
        elif method == 'B':
            d = -np.dot(B, grad)
        else:
            raise ValueError("Invalid method. Choose 'B_inv' or 'B'.")
        
        alpha = armijo_wolfe_line_search(f, gradient_f, x, d, r1)
        
        x = x + alpha * d
        beta1 *= r
        
        iter_count += 1
        func_evals += 1  # f is evaluated in the line search
        grad_evals += 1  # gradient is evaluated once per iteration
    
    return x, iter_count, func_evals, grad_evals


r1 = 8  
x0 = np.array([0.5, 0.25])
beta1 = 1e-4
r = 0.5
epsilon = 10**(-3.5)
max_iter = 10000

# Run the descent method with B^(-1)
result_B_inv, iterations_B_inv, func_evals_B_inv, grad_evals_B_inv = descent_method(x0, r1, beta1, r, epsilon, max_iter, 'B_inv')

print("Results for d^k = -B^(-1)∇f(x^k):")
print(f"Number of iterations: {iterations_B_inv}")
print(f"Number of function evaluations: {func_evals_B_inv}")
print(f"Number of gradient evaluations: {grad_evals_B_inv}")
print(f"Final solution: {result_B_inv}")
print(f"Final function value: {f(result_B_inv, r1)}")
print(f"Final gradient norm: {np.linalg.norm(gradient_f(result_B_inv, r1))}")

print("\n" + "="*50 + "\n")

# Run the descent method with B
result_B, iterations_B, func_evals_B, grad_evals_B = descent_method(x0, r1, beta1, r, epsilon, max_iter, 'B')

print("Results for d^k = -B∇f(x^k):")
print(f"Number of iterations: {iterations_B}")
print(f"Number of function evaluations: {func_evals_B}")
print(f"Number of gradient evaluations: {grad_evals_B}")
print(f"Final solution: {result_B}")
print(f"Final function value: {f(result_B, r1)}")
print(f"Final gradient norm: {np.linalg.norm(gradient_f(result_B, r1))}")

print("\n" + "="*50 + "\n")

print("Comparison:")
print(f"B^(-1) method takes {'less' if iterations_B_inv < iterations_B else 'more'} iterations than B method.")


# ### Question 3
# ![image.png](attachment:image.png)

# In[23]:


def rosenbrock(x):
    return sum(100 * (x[2*i+1] - x[2*i]**2)**2 + (1 - x[2*i])**2 for i in range(5))

def rosenbrock_gradient(x):
    grad = np.zeros_like(x)
    for i in range(5):
        grad[2*i] = -400 * x[2*i] * (x[2*i+1] - x[2*i]**2) - 2 * (1 - x[2*i])
        grad[2*i+1] = 200 * (x[2*i+1] - x[2*i]**2)
    return grad

def generate_B_matrix():
    B = np.random.uniform(0, 1, (10, 10))
    B = (B + B.T) / 2  # Make it symmetric
    B += np.diag(np.random.uniform(5, 10, 10))  # Add diagonal elements
    return B

def armijo_wolfe_line_search(f, grad_f, x, d, alpha=1.0, c1=1e-4, c2=0.9, max_iter=25):
    phi = lambda alpha: f(x + alpha * d)
    dphi = lambda alpha: np.dot(grad_f(x + alpha * d), d)
    
    phi_0 = phi(0)
    dphi_0 = dphi(0)
    
    for _ in range(max_iter):
        if phi(alpha) <= phi_0 + c1 * alpha * dphi_0 and dphi(alpha) >= c2 * dphi_0:
            return alpha
        alpha *= 0.5
    
    return alpha

def optimize(method, max_iter=10000, epsilon=10**(-3.5)):
    x = np.full(10, 0.5)
    B = generate_B_matrix() if method == 'B_inv' else None
    B_inv = np.linalg.inv(B) if method == 'B_inv' else None
    
    for iter in range(max_iter):
        grad = rosenbrock_gradient(x)
        
        if np.linalg.norm(grad) < epsilon:
            break
        
        if method == 'gradient':
            d = -grad
        elif method == 'B_inv':
            d = -np.dot(B_inv, grad)
        
        alpha = armijo_wolfe_line_search(rosenbrock, rosenbrock_gradient, x, d)
        x = x + alpha * d
    
    return x, iter + 1, rosenbrock(x)

# Gradient Descent
x_grad, iters_grad, f_grad = optimize('gradient')
print("Gradient Descent Method:")
print(f"Iterations: {iters_grad}")
print(f"Final function value: {f_grad}")
print(f"Solution: {x_grad}")
print(f"Gradient norm: {np.linalg.norm(rosenbrock_gradient(x_grad))}")

print("\n" + "="*50 + "\n")

# B^(-1) Method
x_B_inv, iters_B_inv, f_B_inv = optimize('B_inv')
print("B^(-1) Method:")
print(f"Iterations: {iters_B_inv}")
print(f"Final function value: {f_B_inv}")
print(f"Solution: {x_B_inv}")
print(f"Gradient norm: {np.linalg.norm(rosenbrock_gradient(x_B_inv))}")

print("\n" + "="*50 + "\n")

print("Comparison:")
print(f"Gradient Descent took {iters_grad} iterations")
print(f"B^(-1) Method took {iters_B_inv} iterations")
print(f"The {'Gradient Descent' if iters_grad < iters_B_inv else 'B^(-1) Method'} converged faster.")


# ### Question 4
# ![image.png](attachment:image.png)

# In[24]:


def f(x):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - np.exp(0.5 * (np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1])))

def gradient_f(x):
    term1 = 20 * 0.2 * 0.5 / np.sqrt(0.5 * (x[0]**2 + x[1]**2)) * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2)))
    term2 = 0.5 * np.exp(0.5 * (np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1])))
    
    dx1 = term1 * x[0] + term2 * 2*np.pi * np.sin(2*np.pi*x[0])
    dx2 = term1 * x[1] + term2 * 2*np.pi * np.sin(2*np.pi*x[1])
    
    return np.array([dx1, dx2])

def armijo_wolfe_line_search(f, grad_f, x, d, alpha=1.0, c1=1e-4, c2=0.9, max_iter=25):
    phi = lambda alpha: f(x + alpha * d)
    dphi = lambda alpha: np.dot(grad_f(x + alpha * d), d)
    
    phi_0 = phi(0)
    dphi_0 = dphi(0)
    
    for _ in range(max_iter):
        if phi(alpha) <= phi_0 + c1 * alpha * dphi_0 and dphi(alpha) >= c2 * dphi_0:
            return alpha
        alpha *= 0.5
    
    return alpha

def optimize(method, x0, B, max_iter=10000, epsilon=10**(-3.5)):
    x = x0
    B_inv = np.linalg.inv(B) if method == 'B_inv' else None
    
    for iter in range(max_iter):
        grad = gradient_f(x)
        
        if np.linalg.norm(grad) < epsilon:
            break
        
        if method == 'gradient':
            d = -grad
        elif method == 'B_inv':
            d = -np.dot(B_inv, grad)
        
        alpha = armijo_wolfe_line_search(f, gradient_f, x, d)
        x = x + alpha * d
    
    return x, iter + 1, f(x)

# Parameters
R = 98  # Last two digit of roll number
x0 = np.array([-R/10, R/10])
B = np.array([[2.5, -1.1], [-1.1, 4.5]])

# Gradient Descent
x_grad, iters_grad, f_grad = optimize('gradient', x0, B)
print("Gradient Descent Method:")
print(f"Iterations: {iters_grad}")
print(f"Final function value: {f_grad}")
print(f"Solution: {x_grad}")
print(f"Gradient norm: {np.linalg.norm(gradient_f(x_grad))}")

print("\n" + "="*50 + "\n")

# B^(-1) Method
x_B_inv, iters_B_inv, f_B_inv = optimize('B_inv', x0, B)
print("B^(-1) Method:")
print(f"Iterations: {iters_B_inv}")
print(f"Final function value: {f_B_inv}")
print(f"Solution: {x_B_inv}")
print(f"Gradient norm: {np.linalg.norm(gradient_f(x_B_inv))}")

print("\n" + "="*50 + "\n")

print("Comparison:")
print(f"Gradient Descent took {iters_grad} iterations")
print(f"B^(-1) Method took {iters_B_inv} iterations")
print(f"The {'Gradient Descent' if iters_grad < iters_B_inv else 'B^(-1) Method'} converged faster.")

