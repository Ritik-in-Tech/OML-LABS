#!/usr/bin/env python
# coding: utf-8

# ### important library imports
# 

# In[206]:


import numpy as np
import pandas as pd
from cvxopt import solvers


# ### Helper function to check linear independence
# 

# In[207]:


def check_linear_independence(matrix):
    rank = np.linalg.matrix_rank(matrix)
    num_columns = matrix.shape[1]
    if rank == num_columns:
        return True
    else:
        return False


# ### Question 1 (part a)
# 

# In[208]:


R=98
matrix=np.array([
    [2,3,1,-2],
    [4,1,1,-5],
    [6,-1,1,-9],
    [2,3,1,R/10],
    [9,8,1,-11],
    [-3,11,1,-13]
])

if check_linear_independence(matrix):
    print("Column vectors are linearly independent")
else:
    print("Column vectors are linearly dependent")


# ### Helper function to check the linear independence from the csv files given
# 

# In[209]:


def check_linear_independence_from_csv(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    matrix = df.to_numpy()
    ans=check_linear_independence(matrix)
    return ans


# ### Question 1 (part b)
# 

# In[210]:


file_path="./A1.csv"
if check_linear_independence_from_csv(file_path):
    print("Column vectors are linearly independent.")
else:
    print("The column vectors are linearly dependent.")


# ### Question 1 (part c)
# 

# In[211]:


data = pd.read_csv("./A2.csv")
A2 = data.to_numpy()
det = np.linalg.det(np.dot(A2.transpose(),A2))
# print("det =",det)
if(det>1e-5):
  print("The column vectors are linearly independent")
else:
  print("The column vectors are linearly dependent")


# ### Question 1 (part d)
# 

# In[212]:


file_path=("./A3.csv")
if check_linear_independence_from_csv(file_path):
    print("Column vectors are linearly independent.")
else:
    print("The column vectors are linearly dependent.")


# ### Helper function to solve the minimization problem of question 2
# 

# In[213]:


def solve_least_squares(A, b):
    # Solve the normal equation A^T A x = A^T b
    ATA = np.dot(A.T, A)
    ATb = np.dot(A.T, b)
    
    try:
        x = np.linalg.solve(ATA, ATb)
        return x
    except np.linalg.LinAlgError:
        return "Matrix is singular, solution cannot be found."


# ### Question 2 (part 1)
# 

# In[214]:


r = 8
A = np.array([[1, r], [2, r], [3, r]])
b = np.array([2, 3, 5])
x_solution = solve_least_squares(A,b)
print("The solution x is:", x_solution)


# ### Question 2 (part 2)
# 

# In[215]:


df=pd.read_csv("./Ab1.csv")
A=df.iloc[:,:-1].values
b=df.iloc[:,-1].values
x_solution=solve_least_squares(A,b)
print("The solution x is:", x_solution)


# ### Question 2 (part 3)
# 

# In[216]:


df=pd.read_csv("./Ab2.csv")
A = df.iloc[:, 0:6].values  
b = df.iloc[:, 6].values
x_solution=solve_least_squares(A,b)
print("The solution x is:", x_solution)


# ### Question 2 (part 4)
# 

# In[217]:


df=pd.read_csv("./Ab3.csv")
A=df.iloc[:,:-1].values
b=df.iloc[:,-1].values
x_solution=solve_least_squares(A,b)
# print(x_solution)
print("The solution x is:", x_solution)


# ### Question 3
# 

# In[218]:


from cvxopt import matrix
P = matrix([[2.0, 0.0], [0.0, 2.0]])
q = matrix([-5.0, -7.0])
G = matrix([[4.0, 4.0, -1.0, 0.0], [1.0, 4.0, 0.0, -1.0]])
h = matrix([20.0, 20.0, 0.0, 0.0])
solution = solvers.qp(P, q, G, h)
x1 = solution['x'][0]
x2 = solution['x'][1]
print(f"Optimal x1: {x1}")
print(f"Optimal x2: {x2}")
objective_value = -x1**2 + 5*x1 - x2**2 + 7*x2 - 5
print(f"Final value of the objective function: {objective_value}")


# ### Question 4
# 

# In[219]:


P = matrix([[1.0, 0.0], [0.0, 0.0]])
q = matrix([0.0, 1.0])
G = matrix([[-1.0, 2.0, 3.0], [-2.0, 5.0, 4.0]])
h = matrix([-15.0, 100.0, 80.0])
solution = solvers.qp(P, q, G, h)
x1 = solution['x'][0]
x2 = solution['x'][1]
print(f"Optimal x1: {x1}")
print(f"Optimal x2: {x2}")
objective_value = 0.5 * x1**2 + x2
print(f"Final value of the objective function: {objective_value}")


# ### Question 5 (part 1)
# 

# In[220]:


file_path = './Ab1.csv'
data = pd.read_csv(file_path, header=None, skiprows=1).values
A = data[:, :-1]
b = data[:, -1]
R = 98.0
P = matrix(A.T @ A)
q = matrix(-A.T @ b)
G = matrix(np.ones((1, A.shape[1])))
h = matrix([R])
solution = solvers.qp(P, q, G, h)


x_optimal = np.array(solution['x'])


print(f"Optimal x: {x_optimal}")

objective_value = 0.5 * x_optimal.T @ (A.T @ A) @ x_optimal + (-A.T @ b).T @ x_optimal
print(f"Final value of the objective function: {objective_value}")


# ### Question 5 (part 2)
# 

# In[221]:


file_path = './Ab2.csv'
data = pd.read_csv(file_path, header=None).values
A = data[1:, :6].astype(float)
b = data[1:, 6].astype(float)
P = matrix(A.T @ A)
q = matrix(-A.T @ b)
G = matrix(np.ones((1, A.shape[1])))
h = matrix([R])
solution = solvers.qp(P, q, G, h)
x_optimal = np.array(solution['x'])
print(f"Optimal x: {x_optimal}")
objective_value = 0.5 * x_optimal.T @ (A.T @ A) @ x_optimal + (-A.T @ b).T @ x_optimal
print(f"Final value of the objective function: {objective_value}")

