import pandas as ps
import sklearn as skl
import os as os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import class_preproc
import time
import numpy as np
import open_csv_files as csv
from sklearn.svm import SVC
import argparse as ag
import class_svm as svm
import class_knn

# Get the current working directory
src = os.getcwd()

parser = ag.ArgumentParser()
parser.add_argument('k', metavar = 'N', type = int)
parser.add_argument('split', metavar = 'Split', type = float)
parser.add_argument('const', metavar = 'Flexibility Constant', type = float)
list_args = parser.parse_args()
dict_args = vars(list_args)
k = dict_args['k']
split = dict_args['split']
c = dict_args['const']
t0 = time.time()
data_x = class_preproc.preProc("bank-additional.csv")
#split = float(input("Enter the split you want: "))
#k = int(input("Enter the value of k for KNN analysis: "))
data_x.processDataset(k, split)
print("Cleaned the training data!")
data_test = class_preproc.testProc("bank-additional-test.csv", "bank-additional-train-cleaned.csv")
data_test.processDataset(k)
print("Data Cleaning done!!")
#bootstrapper = class_preproc.Bootstrap("bank-additional-train-cleaned-encoded.csv")
#bootstrapper.bootstrap()
#print("Bootstrapping done!!")
#dropper = class_preproc.DropFeatures("bank-additional-train-cleaned-encoded.csv", "bank-additional-test-cleaned-encoded.csv")
#dropper.dropFeatures(c)
#print("Dropped features!!")
#feature = class_feature.reg_lasso(src + "/bank-additional-train-cleaned-encoded.csv", src + "/bank-additional-test-cleaned-encoded.csv", nu = 0.1, max_iter = 5000, lambda_ = 7)
#feature.regLogistic()
#cols = feature.regLasso()
#print(cols)
#data_x = ps.read_csv(src + "/bank-additional-train-cleaned-encoded.csv")
#data_test = ps.read_csv(src + "/bank-additional-test-cleaned-encoded.csv")
#data_test.drop(cols, axis = 1, inplace = True)
#data_x.drop(cols, axis = 1, inplace = True)
#data_x.to_csv(src + "/bank-additional-train-cleaned-encoded.csv", index = False)
#data_test.to_csv(src + "/bank-additional-test-cleaned-encoded.csv", index = False)
#print("Reduced Features!!")
t1 = time.time()
print(t1 - t0)

model = svm.class_svm(src + "/bank-additional-train-cleaned-encoded.csv", src + "/bank-additional-test-cleaned-encoded.csv")
#model.svm_train('rbf', (-3, 3), (-3, 3))
model.svm_predict(1.1514, 0.001)

model = class_knn.knnPredictor("bank-additional-train-cleaned-encoded.csv", "bank-additional-test-cleaned-encoded.csv", 150, risk = False)
acc = model.knn_score()
print(acc)
#train_x = csv.open_csv(src + "/bank-additional-train-cleaned-encoded.csv", heading = True)
#print(np.shape(train_x))
#train_y = train_x[:, -1]
#train_y = np.array([0 if item < 1 else 1 for item in train_y])
#print(np.shape(train_y))
#test_x = csv.open_csv(src + "/bank-additional-test-cleaned-encoded.csv", heading = True)
#print(np.shape(test_x))
#test_y = test_x[:, -1]
#test_y = np.array([0 if item < 1 else 1 for item in test_y])
##data_libsvm = np.column_stack(train_y, train_x)
#print("We will now commence SVM Training")
#gamma = np.logspace(-3, 3, 50)
#const = np.logspace(-3, 3, 50)
#acc_final = np.empty((len(gamma), len(const)))
#std_final = np.empty((len(gamma), len(const)))
#roc_final = np.empty((len(gamma), len(const)))
#for i in range(0, len(gamma)):
#    for j in range(0, len(const)):
#        param = SVC(C = const[j], kernel = 'rbf', gamma = gamma[i], class_weight = 'balanced')
#        kf = StratifiedKFold(n_splits = 5, shuffle = True)
#        acc = []
#        roc = []
#        for train_index, test_index in kf.split(train_x, train_y):
#           X_train, X_test = train_x[train_index], train_x[test_index]
#           y_train, y_test = train_y[train_index], train_y[test_index]
#           list_x = X_train.tolist()
#           list_y = list(y_train)
#           model = param.fit(list_x, list_y)
#           list_test_x = X_test.tolist()
#           list_test_y = list(y_test)
#           accy = param.predict(list_test_x)
#           accu = param.score(list_test_x, list_test_y)
#           roc.append(roc_auc_score(list_test_y, accy))
#           acc.append(accu)
#        acc_mean = np.mean(acc)
#        acc_std = np.std(acc)
#        acc_final[i, j] = acc_mean
#        roc_final[i, j] = np.mean(roc)
#        std_final[i, j] = acc_std
#file1 = open(src + "/results_accuracy.csv", 'w')
#print("The maximum accuracy is: ")
#print(np.max(acc_final))
#arg_g, arg_c = argmax(np.max(acc_final), acc_final)
#file1.write("Gamma,C,STD. DEV,ROC\n")
#if(np.shape(arg_c) == (1,)):
#    file1.write(str(gamma[arg_g]) + "," + str(const[arg_c]) + "," + str(std_final[arg_g, arg_c]) + "," + str(roc_final[arg_g, arg_c]) + "\n")
#else:
#    for i, j in zip(arg_c, arg_g):
#        file1.write(str(gamma[i]) + "," + str(const[j]) + "," + str(std_final[i, j]) + "," + str(roc_final[i, j]) + "\n")
#file1.close()
#file1 = open(src + "/results_stddev.csv", 'w')
#print("The minimum standard deviation is: ")
#print(np.min(std_final))
#arg_g, arg_c = argmax(np.min(std_final), std_final)
#file1.write("Gamma,C,Accuracy,ROC\n")
#if(np.shape(arg_c) == (1,)):
#    file1.write(str(gamma[arg_g]) + "," + str(const[arg_c]) + "," + str(acc_final[arg_g, arg_c]) + "," + str(roc_final[arg_g, arg_c]) + "\n")
#else:
#    for i, j in zip(arg_c, arg_g):
#        file1.write(str(gamma[i]) + "," + str(const[j]) + "," + str(acc_final[i, j]) + "," + str(roc_final[i, j]) + "\n")
#file1 = open(src + "/results_ROC.csv", 'w')
#print("The maximum ROC is: ")
#print(np.max(roc_final))
#arg_g, arg_c = argmax(np.max(roc_final), roc_final)
#file1.write("Gamma,C,Accuracy,STD. DEV\n")
#if(np.shape(arg_c) == (1,)):
#    file1.write(str(gamma[arg_g]) + "," + str(const[arg_c]) + "," + str(acc_final[arg_g, arg_c]) + "," + str(std_final[arg_g, arg_c]) + "\n")
#else:
#    for i, j in zip(arg_c, arg_g):
#        file1.write(str(gamma[i]) + "," + str(const[j]) + "," + str(acc_final[i, j]) + "," + str(std_final[i, j]) + "\n")
#prob = svm.problem(list(train_y), train_x.tolist())
#for i, j in zip(arg_c, arg_g):
#    param = SVC(C = const[j], kernel = 'rbf', gamma = gamma[i], class_weight = 'balanced')
#    list_x = test_x.tolist()
#    list_y = list(test_y)
#    model = param.fit(train_x, train_y)
#    accy = param.predict(list_x)
#    accu = param.score(list_x, list_y)
#    roc.append(roc_auc_score(list_y, accy))
#    acc.append(accu)
#max_acc_test = np.max(acc)
#cor_roc_test = roc[np.argmax(acc)]
#max_roc_test = np.max(roc)
#cor_acc_test = acc[np.argmax(roc)]
#print("The maximum testing accuracy is: ")
#print(max_acc_test)
#print("The corresponding ROC is: ")
#print(cor_roc_test)
#print("The maximum testing ROC is: ")
#print(max_roc_test)
#print("The corresponding accuracy is: ")
#print(cor_acc_test)
