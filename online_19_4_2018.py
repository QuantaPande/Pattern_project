# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 01:52:35 2018

@author: chint
"""
import numpy as np
import sklearn as skl
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import online_preproc

data_x = online_preproc.preProc("OnlineNewsPopularityReduced.csv")
data_x.splitDataset()
data_x.normalize()
data_train = data_x.data_train.values
data_test = data_x.data_test.values

label_train = data_x.label_train.values
label_test = data_x.label_test.values

clf = Perceptron(max_iter=5000)
clf.fit(data_train, label_train)
print('Training accuracy is ', clf.score(data_train, label_train))
print('Testing accuracy is ', clf.score(data_test, label_test))

lr = LogisticRegression()
lr.fit(data_train, label_train)
print('Training accuracy lin regression is ', lr.score(data_train, label_train))
print('Testing accuracy lin regression is ', lr.score(data_test, label_test))

pca = PCA(0.995)
pca.fit(data_train)
data_train_pca = pca.transform(data_train)
data_test_pca = pca.transform(data_test)

clf_pca = Perceptron(max_iter=5000)
clf_pca.fit(data_train_pca, label_train)
print('Training accuracy pca is ', clf_pca.score(data_train_pca, label_train))
print('Testing accuracy pca is ', clf_pca.score(data_test_pca, label_test))

lr_pca = LogisticRegression()
lr_pca.fit(data_train_pca, label_train)
print('Training accuracy lin regression pca is ', 
      lr_pca.score(data_train_pca, label_train))
print('Testing accuracy lin regression pca is ', 
      lr_pca.score(data_test_pca, label_test))

knn = KNeighborsClassifier()
knn.fit(data_train, label_train)
print('Training accuracy knn is ', knn.score(data_train, label_train))
print('Testing accuracy knn is ', knn.score(data_test, label_test))

knn_pca = KNeighborsClassifier()
knn_pca.fit(data_train_pca, label_train)
print('Training accuracy knn pca is ', knn_pca.score(data_train_pca, label_train))
print('Testing accuracy knn pca is ', knn_pca.score(data_test_pca, label_test))