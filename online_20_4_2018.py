# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 01:52:35 2018

@author: chint
"""
import numpy as np
import sklearn as skl
import get_data as gd
import online_classifier

answer = input("Is this the first time running code? Enter y if yes, n if no: ")
data_train, label_train, data_test, label_test = gd.get_data(answer)

clf = online_classifier.Classifier()

clf.Ptron(data_train, label_train, data_test, label_test)
clf.LogReg(data_train, label_train, data_test, label_test)
clf.KNN(data_train, label_train, data_test, label_test)
clf.GNB(data_train, label_train, data_test, label_test)

data_train_pca, data_test_pca = clf.pca(data_train, data_test)
clf.pcaPtron(data_train_pca, label_train, data_test_pca, label_test)
clf.pcaLogReg(data_train_pca, label_train, data_test_pca, label_test)
clf.pcaKNN(data_train_pca, label_train, data_test_pca, label_test)
clf.pcaGNB(data_train_pca, label_train, data_test_pca, label_test)

clf.SVM(data_train, label_train, data_test, label_test)
