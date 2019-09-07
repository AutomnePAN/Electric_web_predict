#!/usr/bin/env python
# coding: utf-8

# # Training and Analysing the Data

# ### Import the data set

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sk
import scipy.stats
import matplotlib as plt
import xlrd
import datetime
from sklearn.neural_network import MLPRegressor  # 多层线性回归
from sklearn.preprocessing import minmax_scale
from sklearn import svm
from pandas._libs import tslibs
from sklearn.model_selection import GridSearchCV
import random


# In[2]:


#Import the data_set
raw_data = pd.read_csv('TrainingData/Data_month_with_features_2.csv')


# In[3]:


#The parameters set to test with the MLP
para_set = [[],[],[]]
for a in [10,20,30,40,50,60,70,80,90]:
    for b in [40,50,60,70,80,90,100]:
        para_set[0].append((a,b))
for a in [10,20,30,40,50,60,70,80,90]:
    for b in [40,50,60,70,80,90,100]:
        for c in [40,50,60,70,80,90,100]:
            para_set[1].append((a,b,c))
for a in [10,20,30,40,50,60,70,80,90]:
    for b in [40,50,60,70,80,90,100]:
        for c in [40,50,60,70,80,90,100]:
            for d in [40,50,60,70,80,90,100]:
                para_set[2].append((a,b,c,d))
                    


# ### Set the Training Set

# In[12]:


dataMat = np.array(raw_data)
X = dataMat[51:np.size(dataMat,0)-1,4:]
y = dataMat[51:np.size(dataMat,0)-1,2]
X_sc = minmax_scale(X)
X_train = X_sc[:10000,:]
y_train = y[:10000]
X_test = X_sc[10000:,:]
y_test = y[10000:]


# ## Start Training 

# In[ ]:


best_score = 0
best_para = (0,0,0)
counter = 0
good_para = []
for para in para_set[1]:
    print(para)
    clf  = MLPRegressor(hidden_layer_sizes=para, activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                        learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=2000000, shuffle=True, 
                        random_state=None, tol=1e-8, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                        early_stopping=False)
    clf.fit(X_train,y_train)
    score = clf.score(X_test, y_test)
    if score > 8.80:
        good_para.append(para);
    if score > best_score:
        best_score = score
        best_predict = clf.predict(X_test)
        best_clf = clf
        best_para = para
    counter = counter + 1
#     print(para)
    print('得分：%.3f' %score)
    print('当前最高的分：%.3f' %best_score)
    print(counter, len(para_set))


# ## Analysing the Result

# In[12]:


y_predict = best_predict


# In[84]:


KL = scipy.stats.entropy(y_test,y_predict)
print(KL)


# In[73]:


cengindex = 0
for wi in best_clf.coefs_:
    cengindex += 1  
    print('第%d层网络层:' % cengindex)
    print('权重矩阵维度:',wi.shape)
    print('系数矩阵：\n',wi)


# In[13]:


Y_predict = pd.DataFrame(y_predict)
Y_test = pd.DataFrame(y_test )

