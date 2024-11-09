#!/usr/bin/env python
# coding: utf-8

# ### Question 1
# 

# ### Helper Code
# 

# In[22]:


import numpy as np
import pandas as pd
r='8' # last digit of roll number (B21CS098)
data = pd.read_excel('./2_col_revised.xlsx', header=None, names=['x', 'y'])


if 'r' in str(data.at[99, 'x']):   
    data.at[99, 'x'] = str(data.at[99, 'x']).replace('r', r) 

 
data['x'] = pd.to_numeric(data['x'], errors='coerce')

 
x = data['x'].values
y = data['y'].values
N = len(x)  

beta_1, beta_2 = np.random.rand(2)
learning_rate = 0.01


# In[23]:


def loss_function(beta_1, beta_2, x, y):
    predictions = beta_1 * x + beta_2
    return (1 / (2 * N)) * np.sum((predictions - y) ** 2)


def compute_gradients(beta_1, beta_2, x, y):
    predictions = beta_1 * x + beta_2
    error = predictions - y
    gradient_beta_1 = (1 / N) * np.sum(error * x)
    gradient_beta_2 = (1 / N) * np.sum(error)
    return gradient_beta_1, gradient_beta_2

 
tolerance = 0.01
while True:
     
    grad_beta_1, grad_beta_2 = compute_gradients(beta_1, beta_2, x, y)
    
    
    beta_1 -= learning_rate * grad_beta_1
    beta_2 -= learning_rate * grad_beta_2
    
    
    gradient_magnitude = np.sqrt(grad_beta_1 ** 2 + grad_beta_2 ** 2)
    
     
    if gradient_magnitude < tolerance:
        break

 
print("Optimal β1 (slope):", beta_1)
print("Optimal β2 (intercept):", beta_2)

 
final_loss = loss_function(beta_1, beta_2, x, y)
print("Final loss:", final_loss)


# ### Question 2
# 

# In[24]:


def loss_function(beta_1, beta_2, x, y):
    predictions = beta_1 * x + beta_2
    return (1 / (2 * N)) * np.sum((predictions - y) ** 2)

def compute_gradients(beta_1, beta_2, x_i, y_i):
    prediction = beta_1 * x_i + beta_2
    error = prediction - y_i
    gradient_beta_1 = error * x_i
    gradient_beta_2 = error
    return gradient_beta_1, gradient_beta_2

tolerance = 0.01
max_iterations = 1000  

for iteration in range(max_iterations):
     
    random_index = np.random.randint(N)
    x_i = x[random_index]
    y_i = y[random_index]
    
    grad_beta_1, grad_beta_2 = compute_gradients(beta_1, beta_2, x_i, y_i)
    
    beta_1 -= learning_rate * grad_beta_1
    beta_2 -= learning_rate * grad_beta_2
    
    gradient_magnitude = np.sqrt(grad_beta_1 ** 2 + grad_beta_2 ** 2)
    
    if gradient_magnitude < tolerance:
        print(f"Converged after {iteration + 1} iterations.")
        break

print("Optimal β1 (slope):", beta_1)
print("Optimal β2 (intercept):", beta_2)

final_loss = loss_function(beta_1, beta_2, x, y)
print("Final loss:", final_loss)


# ### Question 3
# 

# In[25]:


batch_size = 10  
def loss_function(beta_1, beta_2, x, y):
    predictions = beta_1 * x + beta_2
    return (1 / (2 * N)) * np.sum((predictions - y) ** 2)

 
def compute_gradients(beta_1, beta_2, x_batch, y_batch):
    predictions = beta_1 * x_batch + beta_2
    errors = predictions - y_batch
    gradient_beta_1 = (1 / len(x_batch)) * np.sum(errors * x_batch)
    gradient_beta_2 = (1 / len(x_batch)) * np.sum(errors)
    return gradient_beta_1, gradient_beta_2

 
tolerance = 0.01
max_iterations = 1000  

for iteration in range(max_iterations):
    
    indices = np.random.choice(N, batch_size, replace=False)
    x_batch = x[indices]
    y_batch = y[indices]
    
    
    grad_beta_1, grad_beta_2 = compute_gradients(beta_1, beta_2, x_batch, y_batch)
    
   
    beta_1 -= learning_rate * grad_beta_1
    beta_2 -= learning_rate * grad_beta_2
    
    
    gradient_magnitude = np.sqrt(grad_beta_1 ** 2 + grad_beta_2 ** 2)
    
    
    if gradient_magnitude < tolerance:
        print(f"Converged after {iteration + 1} iterations.")
        break


print("Optimal β1 (slope):", beta_1)
print("Optimal β2 (intercept):", beta_2)


final_loss = loss_function(beta_1, beta_2, x, y)
print("Final loss:", final_loss)


# ### Question 4
# 

# In[26]:


beta_0, beta_1, beta_2 = np.random.rand(3)

learning_rate = 0.001  

def loss_function(beta_0, beta_1, beta_2, x, y):
    predictions = beta_2 * x**2 + beta_1 * x + beta_0
    return (1 / (2 * N)) * np.sum((predictions - y) ** 2)

def compute_gradients(beta_0, beta_1, beta_2, x, y):
    predictions = beta_2 * x**2 + beta_1 * x + beta_0
    error = predictions - y
    gradient_beta_0 = (1 / N) * np.sum(error)
    gradient_beta_1 = (1 / N) * np.sum(error * x)
    gradient_beta_2 = (1 / N) * np.sum(error * x**2)
    return gradient_beta_0, gradient_beta_1, gradient_beta_2

tolerance = 0.01
max_iterations = 1000  

for iteration in range(max_iterations):
    grad_beta_0, grad_beta_1, grad_beta_2 = compute_gradients(beta_0, beta_1, beta_2, x, y)
    beta_0 -= learning_rate * grad_beta_0
    beta_1 -= learning_rate * grad_beta_1
    beta_2 -= learning_rate * grad_beta_2
    
    gradient_magnitude = np.sqrt(grad_beta_0 ** 2 + grad_beta_1 ** 2 + grad_beta_2 ** 2)
    
 
    if gradient_magnitude < tolerance:
        print(f"Converged after {iteration + 1} iterations.")
        break

print("Optimal β0 (intercept):", beta_0)
print("Optimal β1 (linear coefficient):", beta_1)
print("Optimal β2 (quadratic coefficient):", beta_2)

final_loss = loss_function(beta_0, beta_1, beta_2, x, y)
print("Final loss:", final_loss)


# ### Question 5 and Question 6 are same as Question 2 and Question 3
# 

# ### Question 7
# 

# In[27]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('./diabetes.csv')

data['outcome'] = data['outcome'].apply(lambda x: 1 if x == 'TRUE' else 0)   


X = data.drop(columns=['outcome'])
y = data['outcome']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss_function(X, y, beta):
    m = len(y)
    predictions = sigmoid(np.dot(X, beta))
    return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))


def compute_gradients(X, y, beta):
    m = len(y)
    predictions = sigmoid(np.dot(X, beta))
    gradients = np.dot(X.T, (predictions - y)) / m
    return gradients

# (i) Stochastic Gradient Descent (SGD)
def logistic_regression_sgd(X_train, y_train, learning_rate=0.01, tolerance=0.01, max_iterations=100):
    m, n = X_train.shape
    beta = np.zeros(n)  
    for iteration in range(max_iterations):
        for i in range(m):
            xi = X_train[i:i+1]
            yi = y_train.iloc[i:i+1]
            
            gradients = compute_gradients(xi, yi, beta)
            beta -= learning_rate * gradients
        
        loss = loss_function(X_train, y_train, beta)
        if loss < tolerance:
            print(f"SGD converged after {iteration + 1} iterations.")
            break

    return beta

# (ii) Mini-Batch Gradient Descent (with batch size of 10)
def logistic_regression_mini_batch(X_train, y_train, batch_size=10, learning_rate=0.01, tolerance=0.01, max_iterations=100):
    m, n = X_train.shape
    beta = np.zeros(n)   
    for iteration in range(max_iterations):
        indices = np.random.permutation(m)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train.iloc[indices]

        for i in range(0, m, batch_size):
            Xi = X_train_shuffled[i:i+batch_size]
            yi = y_train_shuffled.iloc[i:i+batch_size]
            
            gradients = compute_gradients(Xi, yi, beta)
            beta -= learning_rate * gradients
        
        loss = loss_function(X_train, y_train, beta)
        if loss < tolerance:
            print(f"Mini-Batch Gradient Descent converged after {iteration + 1} iterations.")
            break

    return beta

beta_sgd = logistic_regression_sgd(X_train, y_train, learning_rate=0.01)


beta_mini_batch = logistic_regression_mini_batch(X_train, y_train, batch_size=10, learning_rate=0.01)


def predict(X, beta):
    return sigmoid(np.dot(X, beta)) >= 0.5


y_pred_sgd = predict(X_test, beta_sgd)
accuracy_sgd = np.mean(y_pred_sgd == y_test)
print(f"SGD Test Accuracy: {accuracy_sgd * 100:.2f}%")


y_pred_mini_batch = predict(X_test, beta_mini_batch)
accuracy_mini_batch = np.mean(y_pred_mini_batch == y_test)
print(f"Mini-Batch Gradient Descent Test Accuracy: {accuracy_mini_batch * 100:.2f}%")

