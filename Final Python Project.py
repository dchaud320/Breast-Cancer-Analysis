#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


#importing our cancer dataset
dataset = pd.read_csv('cancerdata.csv.csv')


# In[3]:


dataset.head()


# In[4]:


print("Cancer data set dimensions : {}".format(dataset.shape))


# In[5]:


dataset.groupby('diagnosis').size()


# In[6]:


dataset.groupby('diagnosis').hist(figsize=(12, 12))


# In[7]:


#extraxting the feature variables and the target variable
X = dataset.iloc[:, 1:31].values
Y = dataset.iloc[:, 31].values


# In[9]:


#Missing or Null Data points
dataset.isnull().sum()
dataset.isna().sum()


# In[10]:


#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# In[11]:


Y


# In[12]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[13]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[14]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)


# In[15]:


Y_pred = classifier.predict(X_test)


# In[16]:


Y_pred


# In[18]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
c = print(cm[0, 0] + cm[1, 1])


# In[20]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test, Y_pred)


# In[21]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
accuracy_score(Y_test, Y_pred)


# In[22]:


#SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train) 
Y_pred = classifier.predict(X_test)
accuracy_score(Y_test, Y_pred)


# In[23]:


#Naives Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
accuracy_score(Y_test, Y_pred)


# In[24]:


#decision tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
accuracy_score(Y_test, Y_pred)


# In[25]:


#Fitting Random Forest Classification Algorithm
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
accuracy_score(Y_test, Y_pred)


# In[ ]:




