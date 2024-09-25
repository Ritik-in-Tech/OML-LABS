#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd


# In[ ]:


# Load the data
data = pd.read_csv("./diabetes2.csv")
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values * 2 - 1   
X = (X - X.mean(axis=0)) / X.std(axis=0)  
X = np.hstack((np.ones((X.shape[0], 1)), X))   

def p(a, x):
    return 1 / (1 + np.exp(-np.dot(a, x)))

def loss(X, y, w):
    return -np.sum(
        np.where(y == 1, np.log(p(w, X.T)), np.log(1 - p(w, X.T)))
    )

def gradient(X, y, w):
    pred = p(w, X.T)
    return -X.T.dot(y - pred)

 
def gradient_descent(X, y, max_iter=10000, tol=1e-3):
    w = np.zeros(X.shape[1])
    
    for i in range(max_iter):
        grad = gradient(X, y, w)
        if np.linalg.norm(grad) < tol:
            break
        
         
        alpha = 1.0
        beta = 0.5
        c = 0.1
        while loss(X, y, w - alpha * grad) > loss(X, y, w) - c * alpha * np.dot(grad, grad):
            alpha *= beta
        
        w -= alpha * grad
    
    return w

 
def mirror_descent(X, y, Q, max_iter=10000, tol=1e-3):
    w = np.zeros(X.shape[1])
    theta = np.zeros(X.shape[1])
    for i in range(max_iter):
        grad = gradient(X, y, w)
        if np.linalg.norm(grad) < tol:
            break
        
        theta -= Q.dot(grad)
        w = np.linalg.solve(Q, theta)
    return w

 
n_features = X.shape[1]
Q = np.diag(np.random.uniform(0, 10, n_features))
Q[np.triu_indices(n_features, 1)] = np.random.uniform(0, 1, (n_features * (n_features - 1)) // 2)
Q = (Q + Q.T) / 2  

 
w_gd = gradient_descent(X, y)
w_md = mirror_descent(X, y, Q)

print("Gradient Descent weights:", w_gd)
print("Mirror Descent weights:", w_md)


# Evaluate models
def accuracy(X, y, w):
    return np.mean((p(w, X.T) >= 0.5) == (y == 1))

print("Gradient Descent accuracy:", accuracy(X, y, w_gd))
print("Mirror Descent accuracy:", accuracy(X, y, w_md))

 

