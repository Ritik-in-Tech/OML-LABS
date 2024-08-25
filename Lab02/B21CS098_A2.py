#!/usr/bin/env python
# coding: utf-8

# ### Helper function and imports
# 

# In[65]:


import numpy as np
from cvxopt import matrix, solvers

def solveMin(c,A,b):
    soln=solvers.lp(matrix(c,tc='d'),matrix(A,tc='d'),matrix(b,tc='d'))
    print(np.round_(soln['x'],decimals=5),soln['primal objective'])


# In[66]:


import numpy as np
from cvxopt import matrix, solvers

def solveMax(c,A,b):
    soln=solvers.lp(matrix(c,tc='d'),matrix(A,tc='d'),matrix(b,tc='d'))
    print(np.round_(soln['x'],decimals=5),-1*soln['primal objective'])


# ### Question 1
# 
# ![image.png](attachment:image.png)
# 

# In[67]:


c=np.array([[3],[-4]])
# print(c)
b=np.array([[12],[20],[-5],[0]])
# print(b)
A=np.array([[1,3],[2,-1],[-1,4],[-1,0]])
solveMin(c,A,b)


# ### Question 2 ( Part a)
# 
# ![image.png](attachment:image.png)
# 

# In[68]:


A=np.array([[1 ,1],[-1,0],[0,-1]])
b=np.array([[150],[-40],[-25]])
c=np.array([[-225],[-200]])
solveMax(c,A,b)


# ### Question 2 (Part b)
# 
# ![image.png](attachment:image.png)
# 

# In[90]:


# Let Xl,Xd be the number of liquid product jars and the number of dry products respectively
# 5Xl + Xd >=10
# 2Xl + 2Xd >=12
# Xl + 4Xd >=12
# Xl , Xd >=0
# min 30Xl + 20Xd 

A  = np.array( [[-5,-1],[-2,-2], [-1,-4],[-1 ,0 ],[0,-1]])
b = np.array( [[-10],[-12],[-12],[0],[0] ])
c = np.array( [ [30],[20] ])

solveMin(c,A,b)


# ### Question 2 (part c)
# 
# ![image.png](attachment:image.png)
# 

# In[88]:


# Let x1,x2,x3,x4,x5 for the parts 1 to 5 respectively
# 2x1+x2+3x3+3x4+x5<=700
# 3x1+2x2+2x3+x4+x5<=1000

# x1,x2,x3,x4,x5>=0

# max 30x1+20x2+40x3+25x4+10x5

A=np.array([[2,1,3,3,1],[3,2,2,1,1],[-1,0,0,0,0],[0,-1,0,0,0],[0,0,-1,0,0],[0,0,0,0,-1],[0,0,0,0,-1]])
b=np.array([[700],[1000],[0],[0],[0],[0],[0]])
c=np.array([[-30],[-20],[-40],[-25],[-10]])

solveMax(c,A,b)


# ### Question 2 part (d)
# 
# ![image.png](attachment:image.png)
# 

# In[87]:


# Let Xa, Xb, Xc be the clothes of type A,B,C
# 2Xa+3Xb <=8
# 2Xb+5Xc <=10
# 2Xa+2Xb+4Xc <=15 
# Xa,Xb,Xc>=0

# max 3Xa+5Xb+4Xc 

A=np.array([[2,3,0],[0,2,5],[3,2,4],[-1,0,0],[0,-1,0],[0,0,-1]])
b=np.array([[8],[10],[15],[0],[0],[0]])
c=np.array([[-3],[-5],[-4]])

solveMax(c,A,b)


# ### Question 3
# 
# ![image.png](attachment:image.png)
# 

# In[70]:


R=98
c = np.array([[R + 3], [R + 41]])
A = np.array([[3, -1], [7, 11],[-1,0],[0,-1]])
b = np.array([[12], [88],[0],[0]])
solveMin(c,A,b)


# ### Question 4
# 
# ![image.png](attachment:image.png)
# 

# In[71]:


R=98
c = np.array([[R], [-2*R + 1]])
A = np.array([[3, -2], [3, -2],[-1,0],[0,-1]])
b = np.array([[1], [6],[0],[0]])
solveMin(c,A,b)


# ### Question 5
# 
# ![image.png](attachment:image.png)
# 

# In[72]:


c=np.array([[-1],[-1],[-1]])
A=np.array([[3,2,1],[2,1,2],[-1,0,0],[0,-1,0],[0,0,-1]])
b=np.array([[3],[2],[0],[0],[0]])

solveMax(c,A,b)


# ### Question 6
# 
# ![image.png](attachment:image.png)
# 

# In[73]:


R=98
A=np.array([[2,1],[1,1],[2,4],[-1,0],[0,-1]])
b=np.array([[1000],[600],[2000],[0],[0]])
c=np.array([[-1*(R+2)],[-1*(R+3)]])

solveMax(c,A,b)


# ### Question 07
# 
# ![image.png](attachment:image.png)
# 

# In[74]:


c = np.array([[2], [3], [10]])
b = np.array([[0], [0], [1], [-1], [0], [0], [0]])
A = np.array([[1, 0, 2], [-1, 0, -2], [0, 1, 1], [0, -1, -1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])

solveMin(c,A,b)


# ### Question 8 is same as Question 1 So I skip this
# 

# ### Question 9
# 
# ![image.png](attachment:image.png)
# 

# In[75]:


R=98
c=np.array([[-1*R],[-1*(R+3)]])
A=np.array([[-3,-1],[-1,-4],[1,1],[-1,0],[0,-1]])
b=np.array([[-3],[-4],[5],[0],[0]])

solveMax(c,A,b)


# ### Question 10
# 
# ![image.png](attachment:image.png)
# 

# In[76]:


R=98
A=np.array([[2,1],[-3,-4],[-2,3],[-1,0],[0,-1]])
b=np.array([[4],[-24],[-6],[0],[0]])
c=np.array([[R],[2*R+1]])
soln=solvers.lp(matrix(c,tc='d'),matrix(A,tc='d'),matrix(b,tc='d'))


# ### Question 11
# 
# ![image.png](attachment:image.png)
# 

# In[77]:


R=98
A=np.array([[1,2],[-4,-3],[3,1],[-3,-1],[-1,0],[0,-1]])
b=np.array([[3],[-6],[3],[-3],[0],[0]])
c=np.array([[R+3],[1]])

solveMin(c,A,b)


# ### Question 12
# 
# ![image.png](attachment:image.png)
# 

# In[78]:


A=np.array([[1,2],[-1,-2],[2,-2],[-2,2],[-1,0],[0,-1]])
b=np.array([[0],[0],[9],[-9],[0],[0]])
c=np.array([[-3],[1]])

# solveMin(c,A,b)
soln=solvers.lp(matrix(c,tc='d'),matrix(A,tc='d'),matrix(b,tc='d'))

