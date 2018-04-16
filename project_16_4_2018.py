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

data_train = data_train.values
print(data_train[270,14])