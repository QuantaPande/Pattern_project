# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 09:41:21 2018

@author: chint
"""

import pandas as ps
import sklearn as skl
from sklearn.preprocessing import OneHotEncoder
import numpy as np

data_train = ps.read_csv("bank-additional.csv")
data_train['y_class'] = data_train['y'].apply(lambda x:0 if x == 'no' else 1)
data_train.drop(['y'], axis = 1, inplace = True)
data_train = data_train.replace(to_replace = 'unknown', value = 'NaN')

data_test = ps.DataFrame(columns = list(data_train))
print(data_train.shape)

for i in range(0, data_train.shape[0]):
    rand = np.random.random_sample()
    if(rand < 0.1):
        data_test = data_test.append(data_train.loc[i, :])
        data_train.drop(i, axis = 0)
data_train.set_index(np.arange(0, data_train.shape[0]))
data_test.set_index(np.arange(0, data_test.shape[0]))

train_knn = ps.DataFrame(columns = list(data_train))
train_knn = train_knn.append(data_train.loc[0, :])
fill_unknown = ps.DataFrame(columns = list(data_train))
fill_unknown = fill_unknown.append(data_train.loc[0, :])
known_index = []
for i in range(1, data_train.shape[0]):
    if (~data_train[i,:] == 'NaN'):
        train_knn = train_knn.append(data_train.loc[i, :])
        known_index.append(i)
    else:
        fill_unknown = fill_unknown.append(data_train.loc[i, :])


train_knn.drop('job', 'maital', 'education', 'default', 'housing', 'loan')
fill_unknown.drop('job', 'maital', 'education', 'default', 'housing', 'loan')

ohe = OneHotEncoder()
for i in range (1,4):
    ohe.fit_transform(train_knn[:,i])
    ohe.fit_transform(fill_unknown[:,i])
ohe.fit_transform(train_knn[:,7])
ohe.fit_transform(fill_unknown[:,7])

train_knn.drop('contact', 'month', 'day_of_week', 'poutcome')
fill_unknown.drop('contact', 'month', 'day_of_week', 'poutcome')


train_knn = train_knn.values
fill_unknown = fill_unknown.values