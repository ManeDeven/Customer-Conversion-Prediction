#!/usr/bin/env python
# coding: utf-8

# In[170]:


#importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[171]:


#ignore warnings
import warnings
warnings.filterwarnings("ignore")  


# In[172]:


#Read the CSV File
df = pd.read_csv('train.csv')
df


# In[173]:


vc_df = pd.DataFrame(df['y'].value_counts()).reset_index()
vc_df.columns = ['Ans', 'Count']
vc_df['Ans'] = vc_df['Ans'].map({1 : 'yes', 0: 'no'})
vc_df


# **From above result we can understand it is imbalanced Dataset**

# # DATA CLEANING

# In[174]:


df.columns


# In[175]:


#checking for Duplicates
df.duplicated().sum()


# In[176]:


#droping duplicates
df = df.drop_duplicates()

df.duplicated().sum()


# In[177]:


#checking for null values
df.isnull().sum()


# **There are No Null Values in Data** 

# In[178]:


# Check tha data Types
df.dtypes


# ### Categorical Data

# In[179]:


df['job'].value_counts()


# Replacing the unknown value 

# In[180]:


df['job'] = df['job'].replace('unknown', method='bfill')
df['job'].value_counts()


# In[181]:


df['marital'].value_counts()


# In[182]:


df['education_qual'].value_counts()


# Replacing the Unknown Value

# In[183]:


df['education_qual'] = df['education_qual'].replace('unknown', method='bfill')
df['education_qual'].value_counts()


# In[184]:


df['call_type'].value_counts()


# In[185]:


df['mon'].value_counts()


# In[186]:


df['prev_outcome'].value_counts()


# ### Continous Data

# In[187]:


#Outlier Detuction using Box Plot for Age Column
sns.set(style="whitegrid")
sns.boxplot(x=df['age'], color='green')


# From outlier we can see that there are many dots are displayed outside whisker

# In[189]:


q1,q3=np.percentile(df["age"],[25,75])
IQR=q3-q1
upper=q3+1.5*IQR
lower=q1-1.5*IQR
print("Upper bound:",upper,"Lower bound :", lower)


# **Removing the Outlier for age**

# In[190]:


df.age = df.age.clip(10.5,70.5)
sns.set(style="whitegrid")
sns.boxplot(x=df['age'], color='green')


# In[191]:


sns.set(style="whitegrid")
sns.boxplot(x=df['day'], color='green')


# In[192]:


sns.set(style="whitegrid")
sns.boxplot(x=df['dur'], color='green')


# In[193]:


q1,q3=np.percentile(df["dur"],[25,75])
IQR=q3-q1
upper=q3+1.5*IQR
lower=q1-1.5*IQR
print("Upper bound:",upper,"Lower bound :", lower)


# **Removing the Outlier for dur**

# In[194]:


df.dur = df.dur.clip(-221.0,643.0)
sns.set(style="whitegrid")
sns.boxplot(x=df['dur'], color='green')


# In[195]:


sns.set(style="whitegrid")
sns.boxplot(x=df['num_calls'], color='green')


# In[196]:


q1,q3=np.percentile(df["num_calls"],[25,75])
IQR=q3-q1
upper=q3+1.5*IQR
lower=q1-1.5*IQR
print("Upper bound:",upper,"Lower bound :", lower)


# **Removing the Outlier for num_calls**

# In[197]:


df.num_calls = df.num_calls.clip(-2.0,6.0)
sns.set(style="whitegrid")
sns.boxplot(x=df['num_calls'], color='green')


# # EDA - Exploratory Data Analysis

# In[198]:


df['target'] = df["y"].map({"yes":1 , "no": 0})
df


# Adding a Target Column

# ## Categorical Variable vs Target Variable

# In[199]:


plt.figure(figsize=(20,35), dpi=180)
#plt.suptitle("Categorical Data Vs Target", fontsize=20, fontweight='bold', color='maroon')
#Jobs vs Target
plt.subplot(3,3,1)
my_colors = ['Magenta', 'cyan']
sns.countplot(x='job',hue='y',data=df, palette=my_colors)
plt.xticks(rotation=50)
plt.title('Jobs vs Target', fontweight='bold', color='maroon')
plt.xlabel('Job', color='DarkGreen')
plt.ylabel('y', color='DarkGreen')

#Marital Status vs Target
plt.subplot(3,3,2)
my_colors = ['Magenta', 'cyan']
sns.countplot(x='marital',hue='y',data=df, palette=my_colors)
plt.xticks(rotation=50)
plt.title('Marital Status vs Target', fontweight='bold', color='maroon')
plt.xlabel('Marital Status', color='DarkGreen')
plt.ylabel('y', color='DarkGreen')

#Educational Qualification vs Target
plt.subplot(3,3,3)
my_colors = ['Magenta', 'cyan']
sns.countplot(x='education_qual',hue='y',data=df, palette=my_colors)
plt.xticks(rotation=50)
plt.title('Educational Qualification vs Target', fontweight='bold', color='maroon')
plt.xlabel('Educational Qualification', color='DarkGreen')
plt.ylabel('y', color='DarkGreen')

#Month vs Target
plt.subplot(3,3,4)
my_colors = ['Magenta', 'cyan']
sns.countplot(x='mon',hue='y',data=df, palette=my_colors)
plt.xticks(rotation=50)
plt.title('Month vs Target', fontweight='bold', color='maroon' )
plt.xlabel('Month', color='DarkGreen')
plt.ylabel('y', color='DarkGreen')

#Previous Outcome vs Target
plt.subplot(3,3,5)
my_colors = ['Magenta', 'cyan']
sns.countplot(x='prev_outcome',hue='y',data=df, palette=my_colors)
plt.xticks(rotation=50)
plt.title('Previous Outcome vs Target', fontweight='bold', color='maroon' )
plt.xlabel('Previous Outcome', color='DarkGreen')
plt.ylabel('y', color='DarkGreen')

#Call Type vs Target
plt.subplot(3,3,6)
my_colors = ['Magenta', 'cyan']
sns.countplot(x='call_type',hue='y',data=df, palette=my_colors)
plt.xticks(rotation=50)
plt.title('Call Type vs Target', fontweight='bold', color='maroon')
plt.xlabel('Call Type', color='DarkGreen')
plt.ylabel('y', color='DarkGreen')

plt.show()

     


# **Jobs vs Target**
# * Target (No) : Blue Collar
# * Suscribed (Yes): Management
# 
# **Marital Status vs Target**
# * Target (No) : Married
# * Subscribed (Yes): Married
# 
# **Educational Qualification vs Target**
# * Target (No): Secondary
# * Subscribed (Yes): Secondary
# 
# **Month vs Target**
# * Target (No): May
# * Subscribed (Yes): May
# 
# **Previous Outcome vs Target**
# * Target (No): unknown
# * Subscribed (Yes): unknown
# 
# **Call Type vs Target**
# * Target (No): Cellular
# * Subscribed (Yes): Cellular

# ## Categorical Variable vs Target

# In[135]:


plt.figure(figsize=(20, 15), dpi=150)
#sub title to show title for overall plot 
plt.suptitle("Numerical Data Vs Target", fontsize=18,  fontweight='bold', color='maroon') 

#Age vs Target
plt.subplot(2,2,1)
my_colors = ['Magenta', 'DarkBlue']
sns.histplot(x='age',hue='y',data=df, palette=my_colors)
plt.xticks(rotation=50)
plt.title('Age vs Target', fontweight='bold', color='maroon' )
plt.xlabel('Age', color='DarkGreen')
plt.ylabel('y', color='DarkGreen')
#df[['age','target']].corr()

#Day vs Target
plt.subplot(2,2,2)
my_colors = ['Magenta', 'DarkBlue']
sns.histplot(x='day',hue='y',data=df, palette=my_colors)
plt.xticks(rotation=50)
plt.title('Day vs Target', fontweight='bold', color='maroon' )
plt.xlabel('Day', color='DarkGreen')
plt.ylabel('y', color='DarkGreen')
#df[['day','target']].corr()

#Duration vs Target
plt.subplot(2,2,3)
my_colors = ['Magenta', 'DarkBlue']
sns.histplot(x='dur',hue='y',data=df, palette=my_colors)
plt.xticks(rotation=50)
plt.title('Duration vs Target', fontweight='bold', color='maroon' )
plt.xlabel('Duration', color='DarkGreen')
plt.ylabel('y', color='DarkGreen')

#No of Calls vs Target
plt.subplot(2,2,4)
my_colors = ['Magenta', 'DarkBlue']
sns.histplot(x='num_calls',hue='y',data=df, palette=my_colors)
plt.xticks(rotation=50)
plt.title('No of Calls vs Target', fontweight='bold', color='maroon' )
plt.xlabel('No Of Calls', color='DarkGreen')
plt.ylabel('y', color='DarkGreen')

plt.show()


# **Age vs Target**
# * Target : Middle age people
# * Subscribed : Middle age people
# 
# **Day vs Target**
# * Target : Middle of Month
# * Subscribed : Middle of Month
# 
# **Duration vs Target**
# * Duration of call is also important to subscribe for insurance.
# 
# **No of Calls vs Target**
# * No of calls increase subscrition also getting increase.

# ## Encoding

# In[136]:


df.columns


# In[137]:


df['job']=df['job'].map({'blue-collar':1,'entrepreneur':2,'services':3,'housemaid':4,'technician':5,'self-employed':6,'admin.':7,'management':8, 'unemployed':9, 'retired': 10, 'student' : 11})   

df['marital'] =df['marital'].map({'married': 1, 'divorced': 2, 'single' : 3})

df['education_qual'] = df['education_qual'].map({'primary': 1, 'secondary': 2, 'tertiary' :3})

df['mon']=df['mon'].map({'may': 1, 'jul' : 2, 'jan': 3, 'nov': 4, 'jun' : 5, 'aug' : 6, 'feb' : 7, 'apr' : 8, 'oct' : 9, 'dec' : 10 , 'sep': 11, 'mar': 12})

df['call_type'] = df['call_type'].map({'unknown': 1, 'telephone' : 2, 'cellular' : 3})

df['prev_outcome']=df['prev_outcome'].map({'unknown' : 1, 'failure' : 2, 'other' : 3, 'success': 4})

df.head()


# ### Feature and Target Selection

# In[138]:


# X --> Feature y-- > Target

x = df[['age', 'job', 'marital', 'education_qual', 'call_type', 'day', 'mon', 'dur', 'num_calls', 'prev_outcome']].values
y=df['target'].values


# ### Spliting

# In[142]:


# splitting the data as train and test

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state = 3 )


# ### Balancing

# In[145]:


#Balancing the data
from imblearn.combine import SMOTEENN 
smt = SMOTEENN(sampling_strategy='all') 
x_train_smt, y_train_smt = smt.fit_resample(x_train, y_train)

print(len(x_train_smt))
print(len(y_train_smt))


# ### Scaling

# In[147]:


#scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_smt)
x_test_scaled = scaler.transform(x_test)


# # Modelling

# ## Logistic Regression

# In[148]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

lr = LogisticRegression()

lr.fit(x_train_scaled,y_train_smt)
lr.score(x_test_scaled,y_test)
     


# In[149]:


y_pred=lr.predict_proba(x_test_scaled)
y_pred


# In[150]:


log_reg_auroc = roc_auc_score(y_test,y_pred[:,1])
print("AUROC score for logistic regression  :  ",round(log_reg_auroc,2))


# ## K-Nearest Neighbour (KNN)

# In[153]:


from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score 
for i in [1,2,3,4,5,6,7,8,9,10,20,30,40,50]: 
  knn= KNeighborsClassifier(i)
  knn.fit(x_train_scaled, y_train_smt) 
  print("K value :", i, "Train Score : ", knn.score(x_train_scaled,y_train_smt), "Cross Value Accuracy :" , np.mean(cross_val_score(knn, x_test_scaled, y_test, cv=10)))


# k=9 is a good cross validation accuracy of 0.896

# In[154]:


knn= KNeighborsClassifier(i)
knn.fit(x_train_scaled, y_train_smt)
print("KNN Score: ",knn.score(x_test_scaled,y_test)) 
print( "AUROC on the sampled dataset : ",roc_auc_score( y_test, knn.predict_proba(x_test)[:, 1]))


# ## Decision Tree

# In[155]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import roc_auc_score


# In[156]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import roc_auc_score 
dt = DecisionTreeClassifier() 
dt.fit(x_train_smt,y_train_smt) 
print("Decision Tree Score : ", dt.score(x_train_smt,y_train_smt)) 
print( "AUROC on the sampled dataset : ",roc_auc_score( y_test, dt.predict_proba(x_test)[:, 1]))


# In[157]:


from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score #this will help me to do cross- validation
import numpy as np

for depth in [1,2,3,4,5,6,7,8,9,10,20]:
  dt = DecisionTreeClassifier(max_depth=depth) # will tell the DT to not grow past the given threhsold
  # Fit dt to the training set
  dt.fit(x_train_smt, y_train_smt) # the model is trained
  trainAccuracy = accuracy_score(y_train_smt, dt.predict(x_train_smt)) # this is useless information - i am showing to prove a point
  dt = DecisionTreeClassifier(max_depth=depth) # a fresh model which is not trained yet
  valAccuracy = cross_val_score(dt, x_test_scaled, y_test, cv=10) # syntax : cross_val_Score(freshModel,fts, target, cv= 10/5)
  print("Depth  : ", depth, " Training Accuracy : ", trainAccuracy, " Cross val score : " ,np.mean(valAccuracy))


# Depth=4 is a good cross validation accuracy of 0.899

# In[158]:


dt = DecisionTreeClassifier(max_depth=5) 
dt.fit(x_train_smt,y_train_smt) 
print("Decision Tree Score : ", dt.score(x_train_smt,y_train_smt)) 
print( "AUROC on the sampled dataset : ",roc_auc_score( y_test, dt.predict_proba(x_test)[:, 1]))


# ## XG Boost

# In[161]:


import xgboost as xgb
from sklearn.model_selection import cross_val_score 
import numpy as np 
for lr in [0.01,0.02,0.03,0.04,0.05,0.1,0.11,0.12,0.13,0.14,0.15,0.2,0.5,0.7,1]: 
  model = xgb.XGBClassifier(learning_rate = lr, n_estimators=100, verbosity = 0) # initialise the model 
  model.fit(x_train_smt,y_train_smt) #train the model 
  print("Learning rate : ", lr," Train score : ", model.score(x_train_smt,y_train_smt)," Cross-Val score : ", np.mean(cross_val_score(model, x_test, y_test, cv=10)))


# Learning Rate 0.1 is getting the best cross validation score of 0.907

# ## Random Forest

# In[162]:


from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(max_depth=2,n_estimators=100,max_features="sqrt")    #max_depth=log(no of features)
rf.fit(x_train, y_train)
y_pred= rf.predict(x_test)


# In[163]:


#doing cross validation to get best value of max _depth to prevent overfitted model 
from sklearn.model_selection import cross_val_score 
from sklearn.ensemble import RandomForestClassifier
for depth in [1,2,3,4,5,6,7,8,9,10]:
  rf= RandomForestClassifier(max_depth=depth,n_estimators=100,max_features="sqrt")   # will tell the DT to not grow past the given threhsold
  # Fit dt to the training set
  rf.fit(x_train, y_train) # the model is trained
  rf= RandomForestClassifier(max_depth=depth,n_estimators=100,max_features="sqrt")   # a fresh model which is not trained yet
  valAccuracy = cross_val_score(rf, x_train, y_train, cv=10) # syntax : cross_val_Score(freshModel,fts, target, cv= 10/5)
  print("Depth  : ", depth, " Training Accuracy : ", trainAccuracy, " Cross val score : " ,np.mean(valAccuracy))


# Depth = 10 is giving the good cross validation score fo 0.902

# # Solution Statement

# Models are tested, below are the AUROC value of each model
# 
# * Logistic Regression - AUROC Score is 0.9
# * KNN - AUROC Score is 0.498
# * Decision Tree - AUROC Score is 0.88
# * XG Boost - AUROC Score is 0.907
# * Random Forest - AUROC Score is 0.902
# 
# Hence XG Boost is giving the good AUROC Score of 0.907, so XG Boost is the best model for customer convertion prediction

# # Conclusion

# Based on the Feature Importance given by best machine Learning that will predict if a client subscribed to the insurance.
# 
# The client should focused on the top few features of order given below to have them subscribed to the insurance.
# 
# * Duration - Longer the call better influncing the clients
# * Age - Age of the person plays an important role in insurance. Middle age people are targeted more and people who suscribed to insurance also middle age people.
# * Day - People who subscribed to insurance are mostly mid of the month.
# * Month - In the month of may people subscribed to insurance are more.
# * Job - In this blue collar people are targeted more but people who subscribed more are from management job.

# In[ ]:




