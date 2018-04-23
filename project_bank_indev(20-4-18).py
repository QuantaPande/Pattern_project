# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 06:59:39 2018

@author: chint
"""

import pandas as ps
import sklearn as skl
import numpy as np
import class_preproc
from sklearn.linear_model import Perceptron
import time

t0 = time.time()
data_x = class_preproc.preProc("bank-additional.csv")
split = float(input("Enter the split you want"))
k = int(input("Enter the value of k for KNN analysis"))
data_x.processDataset(k, split)
data_train = data_x.data_train.values
#data_test = data_x.data_test
label_train = data_x.label_train.values
#label_test = data_x.label_test
print(data_train.shape)
t1 = time.time()
print(t1 - t0)
