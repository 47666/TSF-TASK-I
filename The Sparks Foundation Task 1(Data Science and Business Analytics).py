#!/usr/bin/env python
# coding: utf-8

# # GRIPDEC20 TASK 1(Data Science and Business Analytics)
# 
# SAHIL TANEJA
# 
# 
# 
# 
# 

# # To Explore Supervised Machine Learning
# 
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied.This is a simple linear regression task as it invoves just two variables.Data can be found at http://bit.ly/w-data

# In[1]:


#Importing necessary Libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Reading Dataset
data =pd.read_csv ("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
data 



# In[3]:


data.shape


# In[4]:


data.describe()


# Here We have to visualize the Data to give a better understanding of the correlation between variables (Since the Dataset is quite small)

# In[5]:


data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Scores Obtained')
plt.show()


# It is clearly seen that there is Positive linear relation between the number of hours studied and the marks obtained

# In[6]:


#Now lets divide our data to independent and depedent variables


# In[7]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:



X_train, X_test, y_train, y_test = train_test_split(X, y, 
                           test_size=0.3, random_state=0)


# # Training the algorithm

# In[10]:


from sklearn.linear_model import LinearRegression  
reg = LinearRegression()  


# In[11]:


reg.fit(X_train,y_train)


# In[12]:


print("Training is completed")


# Now we have to visualize the linear regression(i.e. How line will fit the data )

# In[13]:


line = reg.coef_*X+reg.intercept_


# Plotting for testing data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[14]:


#To retrieve the intercept and coefficient

print("Intercept is :")
print(reg.intercept_)


# In[15]:


print("coefficient is : ")
print(reg.coef_)


# This means that for every one unit of change in hours studied, The change in the score is about 9.78%.Or in simpler words, if a student studies one hour more than they previously studied for an exam,they can expect to achieve an increase of 9.78% in the score achieved by the student previously.
# 
# Now lets first make predictions on testing data

# In[16]:


y_pred = reg.predict(X_test)


# In[17]:


y_pred


# The y_pred is a numpy array that contains all the predicted values for the input values in the X_test series
# 
# To compare the actual output values for X_test with the predicted values, execute the following script

# In[18]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[19]:


# Now lets visualize the predicted and actual values 

plt.scatter(X_test,y_test)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('testing data actual values')
plt.show()

plt.scatter(X_test,y_pred,marker='v')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Testing data predicted values')
plt.show()


# In[20]:


# You can also test with your own data
hours = [[9.25]]
k_pred = reg.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(k_pred[0]))


# # Evaluating Algorithm
# 
# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For regression algorithms, three evaluation metrics are continuouly used
# 
# 1) Mean Squared Error
# 
# 2) Mean Absoulte Error
# 
# 3) Root mean Squared Error

# In[21]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# END OF TASK 1 
