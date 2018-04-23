#import pandas as ps
#import sklearn as skl
#import numpy as np
#import class_preproc
#from sklearn.linear_model import Perceptron
#import time

#t0 = time.time()
#data_x = class_preproc.preProc("bank-additional.csv")
#split = float(input("Enter the split you want: "))
#k = int(input("Enter the value of k for KNN analysis: "))
#data_x.processDataset(k, split)
#data_train = data_x.data_train.values
##data_test = data_x.data_test
#label_train = data_x.label_train.values
##label_test = data_x.label_test
#print(data_train.shape)
#t1 = time.time()
#print(t1 - t0)

import pandas as ps
import sklearn as skl
import os as os
from sklearn.model_selection import StratifiedKFold
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import class_preproc
import time
import numpy as np
import open_csv_files as csv
import svmutil as svm
import argparse as ag

def argmax(max, a):
    n = np.shape(a)
    arg_i = []
    arg_j = []
    for i in range(0, n[0]):
        for j in range(0, n[1]):
            if a[i, j] == max:
                arg_i.append(i)
                arg_j.append(j)
    return arg_i, arg_j
# Get the current working directory
src = os.getcwd()

parser = ag.ArgumentParser()
parser.add_argument('k', metavar = 'N', type = int)
parser.add_argument('split', metavar = 'Split', type = float)
list_args = parser.parse_args()
dict_args = vars(list_args)
k = dict_args['k']
split = dict_args['split']
t0 = time.time()
data_x = class_preproc.preProc("bank-additional.csv")
#split = float(input("Enter the split you want: "))
#k = int(input("Enter the value of k for KNN analysis: "))
data_x.processDataset(k, split)
print("Cleaned the training data!")
data_test = class_preproc.testProc("bank-additional-test.csv", "bank-additional-train-cleaned.csv")
data_test.processDataset(k)
print("Data Cleaning done!!")
t1 = time.time()
print(t1 - t0)

train_x = csv.open_csv(src + "/bank-additional-train-cleaned-encoded.csv", heading = True)
print(np.shape(train_x))
train_y = train_x[:, -1]
train_y = np.array([0 if item < 1 else 1 for item in train_y])
print(np.shape(train_y))
test_x = csv.open_csv(src + "/bank-additional-test-cleaned-encoded.csv", heading = True)
print(np.shape(test_x))
test_y = test_x[:, -1]
test_y = np.array([0 if item < 1 else 1 for item in test_y])
#data_libsvm = np.column_stack(train_y, train_x)
print("We will now commence SVM Training")
gamma = np.logspace(-3, 3, 50)
const = np.logspace(-3, 3, 50)
acc_final = np.empty((len(gamma), len(const)))
std_final = np.empty((len(gamma), len(const)))
for i in range(0, len(gamma)):
    for j in range(0, len(const)):
        param = svm.svm_parameter('-t 2 -g ' + str(gamma[i]) + ' -c ' + str(const[j]) + ' -q')
        kf = StratifiedKFold(n_splits = 5, shuffle = True)
        acc = []
        for train_index, test_index in kf.split(train_x, train_y):
           X_train, X_test = train_x[train_index], train_x[test_index]
           y_train, y_test = train_y[train_index], train_y[test_index]
           list_x = X_train.tolist()
           list_y = list(y_train)
           prob = svm.svm_problem(list_y, list_x)
           model = svm.svm_train(prob, param)
           list_test_x = X_test.tolist()
           list_test_y = list(y_test)
           accy, dummy_1, dummy_2 = svm.svm_predict(list_test_y, list_test_x, model, '-q')
           accu, dummy_1, dummy_2 = svm.evaluations(list_test_y, accy)
           acc.append(accu)
        acc_mean = np.mean(acc)
        acc_std = np.std(acc)
        acc_final[i, j] = acc_mean
        std_final[i, j] = acc_std
print("The maximum accuracy is: ")
print(np.max(acc_final))
print("It is found for values of gamma and C equal to: ")
arg_g, arg_c = argmax(np.max(acc_final), acc_final)
print("Gamma          C           STD. DEV")
if(np.shape(arg_c) == (1,)):
    print(str(gamma[arg_g]) + str(const[arg_c]) + str(std_final[arg_g, arg_c]))
else:
    for i in arg_c:
        for j in arg_g:
            print(str(gamma[i]) + str(const[j]) + str(std_final[i, j]))
print("The minimum standard deviation is: ")
print(np.min(std_final))
print("It is found for values of gamma and C equal to: ")
arg_g, arg_c = argmax(np.max(std_final), std_final)
print("Gamma          C             Accuracy")
if(np.shape(arg_c) == (1,)):
    print(str(gamma[arg_g]) + str(const[arg_c]) + str(acc_final[arg_g, arg_c]))
else:
    for i in arg_c:
        for j in arg_g:
            print(str(gamma[i]) + str(const[j]) + str(acc_final[i, j]))
fig = plt.figure(1)
ax = fig.gca(projection = '3d')

X, Y = np.meshgrid(gamma, const)
surf = ax.plot_surface(X, Y, acc_final, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
ax.set_zlim(np.min(acc_final), np.max(acc_final))
ax.set_xlabel('Gamma')
ax.set_ylabel('C')
ax.set_zlabel('Accuracy')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('figure_' + str(1) + '.png')

fig = plt.figure(2)
ax = fig.gca(projection = '3d')

surf = ax.plot_surface(X, Y, std_final, cmap = cm.coolwarm, linewidth = 0, antialiased = False)
ax.set_zlim(np.min(std_final), np.max(std_final))
ax.set_xlabel('Gamma')
ax.set_ylabel('C')
ax.set_zlabel('Std Deviation')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('figure_' + str(2) + '.png')

