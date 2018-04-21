# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 01:45:03 2018

@author: chint
"""
import pandas as ps
import numpy as np
from sklearn.preprocessing import StandardScaler

class preProc():
    def __init__(self, filename):
        self.data_train = ps.read_csv(filename)
        self.data_train.drop(['url', 'timedelta'], axis = 1, inplace = True)
        self.data_train['y'] = self.data_train['shares'].apply(lambda x:0 if x <= 1600 else 1)
        self.data_test = ps.DataFrame(columns = list(self.data_train))
    
    def splitDataset(self):
        for i in range(0, self.data_train.shape[0]):
            rand = np.random.random_sample()
            if(rand < 0.1):
                self.data_test = self.data_test.append(self.data_train.loc[i, :])
                self.data_train.drop(i, axis = 0, inplace = True)
        
        self.data_train.set_index(np.arange(0, self.data_train.shape[0]))
        self.data_test.set_index(np.arange(0, self.data_test.shape[0]))
        
        self.label_train = self.data_train['y']
        self.data_train.drop(['shares'], axis = 1, inplace = True)
        self.data_train.to_csv("Online-data-train.csv", index = False)
        self.data_train.drop(['y'], axis = 1, inplace = True)
        
        self.label_test = self.data_test['y']
        self.data_test.drop(['shares'], axis = 1, inplace = True)
        self.data_test.to_csv("Online-data-test.csv", index = False)
        self.data_test.drop(['y'], axis = 1, inplace = True)
    
    def normalize(self, data_train, data_test):    
        self.category = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 
                              'data_channel_is_bus', 'data_channel_is_socmed', 
                              'data_channel_is_tech', 'data_channel_is_world', 
                              'weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 
                              'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday', 
                              'weekday_is_sunday', 'is_weekend']
        data_train_category = ps.DataFrame(columns = list(self.category))
        data_train_category[self.category] = data_train[self.category]
        
        data_test_category = ps.DataFrame(columns = list(self.category))
        data_test_category[self.category] = data_test[self.category]
        
        data_train.drop(self.category, axis = 1, inplace = True)
        data_test.drop(self.category, axis = 1, inplace = True)
        
        self.scale = StandardScaler()
        self.scale.fit(data_train.values[:,:])
        self.scale.transform(data_train.values[:,:])
        self.scale.transform(data_test.values[:,:])
        
        
        #self.data_train = (self.data_train-self.data_train.mean(axis=0))/self.data_train.std(axis=0)
        #self.data_test = (self.data_test-self.data_train.mean(axis=0))/self.data_train.std(axis=0)
        
        data_train[self.category] = data_train_category[self.category]
        data_test[self.category] = data_test_category[self.category]
        return data_train, data_test
