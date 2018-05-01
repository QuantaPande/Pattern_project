# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 05:29:44 2018

@author: chint
"""

import pandas as ps
import online_preproc

def get_data(answer):
    data_x = online_preproc.preProc("OnlineNewsPopularityReduced.csv")

    if (answer == 'y'):
        data_x.splitDataset()
        data_train, data_test = data_x.normalize(data_x.data_train, data_x.data_test)
        data_train = data_train.values
        data_test = data_test.values
        
        label_train = data_x.label_train.values
        label_test = data_x.label_test.values
        
    elif (answer == 'n'):
        data_train = ps.read_csv("Online-data-train.csv")
        label_train = data_train['y']
        data_train.drop(['y'], axis = 1, inplace = True)
        
        data_test = ps.read_csv("Online-data-test.csv")
        label_test = data_test['y']
        data_test.drop(['y'], axis = 1, inplace = True)
        data_train, data_test = data_x.normalize(data_train, data_test)
        data_train = data_train.values
        data_test = data_test.values
        
    else:
        print('Please enter valid key')
    
    return data_train, label_train, data_test, label_test