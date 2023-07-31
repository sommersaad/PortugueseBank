#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from collections import Counter


# In[3]:


df = pd.read_csv("bank-additional-full.csv")
df.head(10)


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe().astype(np.int64)


# In[7]:


df.isnull().sum()


# In[8]:


df[['age','campaign','duration','previous']].hist(bins=30, figsize=(20,15))
plt.show()


# In[9]:


sns.distplot(df['age'],
hist=True, kde=int(180/5), color='darkblue' , hist_kws={'edgecolor':'black'}, kde_kws={'linewidth':4})


# In[10]:


sns.distplot(df['campaign'],
hist=True, kde=int(180/5), color='darkblue' , hist_kws={'edgecolor':'black'}, kde_kws={'linewidth':4})


# In[11]:


sns.distplot(df['duration'],
hist=True, kde=int(180/5), color='darkblue' , hist_kws={'edgecolor':'black'}, kde_kws={'linewidth':4})


# In[12]:


sns.distplot(df['previous'],
hist=True, kde=int(180/5), color='darkblue' , hist_kws={'edgecolor':'black'}, kde_kws={'linewidth':4})


# In[13]:


fig, ax = plt.subplots(figsize=(10,10))  
sns.heatmap(df._get_numeric_data().astype(float).corr(),
            square=True, cmap='CMRmap_r', linewidths=.5,
            annot=True, fmt='.2f').figure.tight_layout()
plt.show()


# In[14]:


#clean values that are in catagories to numericals or data that can be called upon 
category_features = df.select_dtypes(include=['object', 'bool']).columns.values
for col in category_features:
    print(col, "(", len(df[col].unique()) , "values):\n", np.sort(df[col].unique()))


# In[15]:


for col in category_features:
    print(f"\033[1m\033[94m{col} \n{20 * '-'}\033[0m")    
    print(df[col].value_counts(), "\n")
    
print(df.nunique(axis=1))


# In[16]:


#visualize data 
for col in category_features:
    plt.figure(figsize=(20,5))    
    sns.barplot(df[col].value_counts().values, df[col].value_counts().index, data=df)    
    plt.title(col)    
    plt.tight_layout()


# In[17]:


df.groupby(['campaign'])['y'].count().reset_index().sort_values(by='y', ascending=False).iloc[:5]


# In[18]:


table = pd.crosstab(df.job, df.y)
table.columns = ['Not subscribed', 'Subscribed']
table.plot(kind='bar')

plt.grid(True)

plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')


# In[19]:


table = pd.crosstab(df.job, df.y)
table = round(table.div(table.sum(axis=1), axis=0).mul(100), 2)
table.columns=['notsubcribed', 'subcribed']
table.sort_values(by=['subcribed'], ascending=False).loc[:, 'subcribed']


# In[20]:


table = pd.crosstab(df.marital,df.y)
table = table.div(table.sum(1).astype(float), axis=0)
table.columns = ['Not subscribed', 'Subscribed']
# Ordering stacked bars and plot the chart
table[['Subscribed', 'Not subscribed']].plot(kind='bar', stacked=True)
plt.title('Frequency of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')


# In[21]:


df = df.drop(['duration'], axis=1)


# In[22]:


X = df.drop(columns=['y'])
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# In[23]:


numeric_features = X_train.select_dtypes(include=['float64', 'int64', 'string']).columns.values
numeric_features = numeric_features[numeric_features != 'y']

category_features = X_train.select_dtypes(include=['object', 'bool']).columns.values

print(numeric_features)
print(category_features)


# In[24]:


def dummify(ohe, x, columns):
    transformed_array = ohe.transform(x)

    # list of category columns
    enc = ohe.named_transformers_['cat'].named_steps['onehot']
    feature_lst = enc.get_feature_names(category_features.tolist())   
    
    cat_colnames = np.concatenate([feature_lst]).tolist()
    all_colnames = numeric_features.tolist() + cat_colnames 
    
    # convert numpy array to dataframe
    df = pd.DataFrame(transformed_array, index = x.index, columns = all_colnames)
    
    return transformed_array, df


# In[25]:


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])


# In[26]:


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


# In[27]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, category_features)])

ohe = preprocessor.fit(X_train)

X_train_t = ohe.transform(X_train)
X_test_t = ohe.transform(X_test)


# In[28]:


X_train_t_array, X_train_t = dummify(ohe, X_train, category_features)
X_test_t_array, X_test_t = dummify(ohe, X_test, category_features)
X_train_t.head()


# In[29]:


X_train_columns = X_train_t.columns
print(X_train_columns)


# In[30]:


Dataframe= X_train_t, y_train, X_test_t, y_test
Array=X_train_t_array, X_test_t_array


# In[31]:


features = pd.get_dummies(df.drop('y', axis = 1)).values
labels = df['y'].replace({'no': 0, 'yes': 1}).values


# In[32]:


rf = RandomForestClassifier(n_estimators = 100, random_state = 202)


# In[ ]:


sm = SMOTE(sampling_strategy = 'not majority', k_neighbors = 50, random_state = 202)
features_res, labels_res = sm.fit_resample(features, labels)


# In[ ]:




