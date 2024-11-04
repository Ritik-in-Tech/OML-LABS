#!/usr/bin/env python
# coding: utf-8

# #### Question 1
# 

# In[9]:


import numpy as np

tolerance = 1e-6  
max_iterations = 100  
x = np.array([0.0, 0.0, 0.0])  
mu = 1.0  

def grad_f(x):
    return np.array([x[0] - 2, x[1] - 2, x[2] - 2])

H_f = np.eye(3)
grad_g = np.array([1.0, 1.0, 1.0])

def g(x):
    return np.sum(x) - 1

for iteration in range(max_iterations):
    grad_L = grad_f(x) + mu * grad_g  
    residual = np.append(grad_L, g(x))
    
    if np.linalg.norm(residual) < tolerance:
        break

    KKT_matrix = np.block([[H_f, grad_g.reshape(-1, 1)], [grad_g.reshape(1, -1), 0]])
    rhs = -residual

    solution = np.linalg.solve(KKT_matrix, rhs)
    delta_x = solution[:3]
    delta_mu = solution[3]

    x += delta_x
    mu += delta_mu

print("Optimal solution x:", x)
print("Optimal Lagrange multiplier mu:", mu)
print("Iterations:", iteration + 1)


# #### Question 2
# 

# In[10]:


import numpy as np

tolerance = 1e-6  
max_iterations = 100  
x = np.array([0.0, 0.0, 0.0])  
mu = 1.0  

def f(x):
    return 100 * ((x[2] - x[1]**2)**2 + (x[1] - x[0]**2)**2) + ((1 - x[0])**2 + (1 - x[1])**2 + (1 - x[2])**2)

def grad_f(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0]**2) - 400 * x[1] * (x[2] - x[1]**2) - 2 * (1 - x[1]),
        200 * (x[2] - x[1]**2) - 2 * (1 - x[2])
    ])

def hessian_f(x):
    return np.array([
        [-400 * (x[1] - 3 * x[0]**2) + 2, -400 * x[0], 0],
        [-400 * x[0], 200 - 400 * (x[2] - 3 * x[1]**2) + 2, -400 * x[1]],
        [0, -400 * x[1], 200 + 2]
    ])

grad_g = np.array([1.0, 1.0, 1.0])

def g(x):
    return np.sum(x) - 1

for iteration in range(max_iterations):
    grad_L = grad_f(x) + mu * grad_g  
    residual = np.append(grad_L, g(x))
    
    if np.linalg.norm(residual) < tolerance:
        break

    KKT_matrix = np.block([[hessian_f(x), grad_g.reshape(-1, 1)], [grad_g.reshape(1, -1), 0]])
    rhs = -residual

    solution = np.linalg.solve(KKT_matrix, rhs)
    delta_x = solution[:3]
    delta_mu = solution[3]

    x += delta_x
    mu += delta_mu

print("Optimal solution x:", x)
print("Optimal Lagrange multiplier mu:", mu)
print("Iterations:", iteration + 1)


# ### Question 3
# 

# In[11]:


import numpy as np
from scipy.optimize import minimize

 
def objective(x):
    x1, x2 = x
    return 2 * x1**2 + 2 * x1 * x2 + 3 * x2**2 - 2 * x1 + 3 * x2

 
def constraint1(x):
    return 3 * x[0] + 2 * x[1] - 6  

def constraint2(x):
    return x[0]   

def constraint3(x):
    return x[1]  

 
def barrier(x, sigma):
    epsilon = 1e-8
    b1 = -np.log(constraint1(x) + epsilon)
    b2 = -np.log(constraint2(x) + epsilon)
    b3 = -np.log(constraint3(x) + epsilon)
    return objective(x) + sigma * (b1 + b2 + b3)

 
sigma = 1.0   
r = 0.5   
tolerance = 1e-4   

 
x0 = np.array([2.0, 0.5])

 
while sigma > tolerance:
    result = minimize(barrier, x0, args=(sigma,), method='Nelder-Mead')
    x0 = result.x
    sigma *= r

 
x_opt = result.x
grad_f = np.array([4 * x_opt[0] + 2 * x_opt[1] - 2, 2 * x_opt[0] + 6 * x_opt[1] + 3])
A = np.array([[3, 2], [1, 0], [0, 1]])
b = grad_f
lambdas = np.linalg.lstsq(A.T, b, rcond=None)[0]

 
lambdas_rounded = np.round(lambdas).astype(int)
lambdas_rounded[lambdas_rounded < 0] = 0  

 
x_opt_rounded = np.round(x_opt).astype(int)
objective_value_rounded = round(objective(x_opt))

print("Optimal solution x :", x_opt_rounded)
print("Objective value at optimal solution:", objective_value_rounded)
print("KKT multipliers :", lambdas_rounded)


# ### Question 4
# 

# In[12]:


import pandas as pd


# #### Implementation of dual svm and prima svm

# In[13]:


from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def dual_svm(X, y, kernel='linear', C=1.0):
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, max_iter=1000))
    model.fit(X, y)
    return model

def primal_svm(X, y):
    model = make_pipeline(StandardScaler(), LinearSVC(dual=False, max_iter=1000))
    model.fit(X, y)
    return model


# ### Dataset 1 'Diabetes.csv'

# In[14]:


data=pd.read_csv('diabetes.csv')
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
model = primal_svm(x, y)
y_pred = model.predict(x)
print(f" Primal SVM Accuracy:", accuracy_score(y, y_pred))

model = dual_svm(x, y, kernel='linear')
y_pred = model.predict(x)
print(f" Dual SVM Accuracy:", accuracy_score(y, y_pred))


# ### Dataset 2 'generated_test.csv'

# In[15]:


data=pd.read_csv('generated_test.csv')
data
x=data.iloc[:-1,:-1]
y=data.iloc[:-1,-1]
model = primal_svm(x, y)
new_data = np.array([[3, 23 / 10]])

predicted_outcome = model.predict(new_data)

print("Primal Predicted outcome:", predicted_outcome[0])

model = dual_svm(x, y, kernel='linear')
y_pred = model.predict(new_data)
print("Dual Predicted outcome:", predicted_outcome[0])


# ### Dataset 3 '4ColumnDa4ColumnDataset.csv'

# In[16]:


data=pd.read_csv('./4ColumnDa4ColumnDataset.csv')
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
model = primal_svm(x, y)
y_pred = model.predict(x)
print(f" Primal SVM Accuracy:", accuracy_score(y, y_pred))

model = dual_svm(x, y, kernel='linear')
y_pred = model.predict(x)
print(f" Dual SVM Accuracy:", accuracy_score(y, y_pred))

