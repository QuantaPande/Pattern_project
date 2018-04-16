# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 23:53:12 2018

@author: chint
"""

import pandas as ps
import numpy as np
import sklearn as skl

data_train = ps.read_csv("bank-additional-cleaned.csv")
data_train.drop(['index'], axis = 1, inplace = True)
numeric = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
category = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
            'contact', 'month', 'day_of_week', 'poutcome']
data_train_numeric = ps.DataFrame(columns = list(numeric))
data_train_numeric[numeric] = data_train[numeric]

data_train_category = ps.DataFrame(columns = list(category))
data_train_category[category] = data_train[category]


data_train_norm = (data_train_numeric-data_train_numeric.mean(axis=0, numeric_only=True))/data_train_numeric.std(axis=0, numeric_only=True)
data_train_norm[category] = data_train_category[category]
data_train_norm = ps.get_dummies(data_train_norm, 
                               prefix = None, 
                               columns=category)
data_train = data_train_norm.values
print(data_train_norm)
