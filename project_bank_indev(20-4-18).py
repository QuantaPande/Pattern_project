# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 06:59:39 2018

@author: chint
"""

import pandas as ps
import sklearn as skl
import numpy as np
import math
import class_preproc
from sklearn.linear_model import Perceptron
#import test_proc

data_x = class_preproc.preProc("bank-additional.csv")
data_x.processDataset(5)
data_train = data_x.data_train.values
#data_test = data_x.data_test
label_train = data_x.label_train.values
#label_test = data_x.label_test
print(data_train.shape)
