#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder


# In[2]:


# ########################    LOADING THE DATASET    ############################
dataset = pd.read_csv("DM project\SETAP PROCESS DATA CORRECT AS FIE2016\setapProcessT1.csv")
#dataset.groupby('SE Process grade').size()



###############################    DATA PREPROCESSING/ DATA CLEANING   #####################################


df_filtered1 = dataset[dataset['SE Process grade'] == 'A']
df_filtered2 = dataset[dataset['SE Process grade'] == 'F']
data1=df_filtered1.head(120)
data2=df_filtered2.head(120)
dataset = pd.concat([data1, data2], ignore_index=True, sort=False)

#dataset = shuffle(dataset)

#dataset = dataset.drop([57,58,59,60,61,62,63,3,4,5,55])
dataset



#dataset


#pd.concat([x1,x2])
# dataset2=df_filtered1.head(90) + df_filtered2.head(70)
# dataset2
                 
#print(rows = dataset.loc["SE Process grade"]) 
#dataset.drop(labels='A', axis=0, index=None, columns="SE Process grade", level=None, inplace=False, errors=’raise’)
#print(dataset.groupby("label").head(20))
#dataset


# In[3]:


dataset.isnull().values.any()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


# shape
dataset.info()
#dataset.isnull().values.any()


# In[5]:


# shape
print(dataset.shape)
# dataset descriptions
#print(dataset.describe())


# In[6]:


# class distribution
dataset.groupby('SE Process grade').size()


# In[7]:



###################################    TEST / TRAIN SPLIT  ############################
names=[]
results=[]

from sklearn.model_selection import train_test_split
# X=train.iloc[:,:84]
# Y=train['SE Process grade']
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2, random_state=None)
# train, test = train_test_split(dataset, test_size=0.2)


array = dataset.values
X = array[:,:-1]   # all the features except the last one is in X
Y = array[:,84]    # only the last one A , F
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.1, random_state=1)


print('Training set:', X_train.shape[0])
print('Testing set:', X_validation.shape[0])




# In[9]:


##############################   Logestic Regression   #################################

lr=LogisticRegression()
lr.fit(X_train,Y_train)
# lr.score(X_train,Y_train)
# print(lr.score(X_train,Y_train))
y_pred=lr.predict(X_validation)
print ("Accuracy : ", accuracy_score(Y_validation, y_pred)) 
print(confusion_matrix(Y_validation, y_pred))
print (classification_report(Y_validation, y_pred)) 
result=accuracy_score(Y_validation, y_pred)
results.append(result)
names.append("LR")


# In[ ]:





# In[10]:


#######################   TAKING ONE INSTANCE TO PREDICT(LABELS) THE GRADE OF STUDENT   ###########################




print(X_validation[6])   #RANDOM ROW TAKEN FROM VALIDATION DATASET
print(Y_validation[6])   #RANDOM ROW LABEL ACCORDING TO DATASET IS
sa=lr.predict(X_validation[6].reshape(1,-1)) #MODEL PREDICTED LABEL
print(sa)


# In[11]:


###########################      Naive Bayes   ################################

gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train).predict(X_validation)
print('Acurracy of training set ',(gnb.score(X_train, Y_train)*100))
print('Acurracy of testing set ',(accuracy_score(Y_validation, y_pred)*100))
print (classification_report(Y_validation, y_pred)) 
results.append(accuracy_score(Y_validation, y_pred))
names.append("GNB")


# In[12]:


###########################          Random Forest Classifier   ################################

from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier()
model.fit(X_train,Y_train)
# model.score(train_feature,test_target)
# print("For Training =  ",train.shape[0])
# print("For Testing =  ",test.shape[0])
# print(model.score(train_feature,test_target))
# results.append(model.score(train_feature,test_target))
# names.append("RFC")

y_pred=model.predict(X_validation)

print('Acurracy of training set ',(model.score(X_train, Y_train)*100))
print('Acurracy of testing set ',(accuracy_score(Y_validation, y_pred)*100))
print ("Accuracy : ", accuracy_score(Y_validation, y_pred)) 
print (classification_report(Y_validation, y_pred))
results.append(accuracy_score(Y_validation, y_pred))
names.append("RFC")


# In[ ]:





# In[13]:


knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
results.append(accuracy_score(Y_validation, y_pred))
names.append("DTC")


# In[ ]:





# In[ ]:




