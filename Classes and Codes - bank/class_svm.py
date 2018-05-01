import sklearn as skl
import os as os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import time
import numpy as np
import open_csv_files as csv
from sklearn.svm import SVC

class class_svm():
    def __init__(self, training_data, testing_data):
        self.train_x = csv.open_csv(training_data, heading = True)
        print(np.shape(self.train_x))
        end = np.shape(self.train_x)[1] - 1
        self.train_y = self.train_x[:, end]
        self.train_y = np.around(self.train_y)
        self.train_y = self.train_y.astype(int)
        print(np.count_nonzero(self.train_y))
        self.test_x = csv.open_csv(testing_data, heading = True)
        print(np.shape(self.test_x))
        self.test_y = self.test_x[:, end]
        self.test_y = np.around(self.test_y)
        self.test_y = self.test_y.astype(int)
        print(np.count_nonzero(self.test_y))
        if (not np.count_nonzero(self.test_y)):
            print("Another Problem!!")
        print(self.test_y)
        self.src = os.getcwd()

    def argmax(self, max, a):
        n = np.shape(a)
        arg_i = []
        arg_j = []
        for i in range(0, n[0]):
            for j in range(0, n[1]):
                if a[i, j] == max:
                    arg_i.append(i)
                    arg_j.append(j)
        arg_k = np.array(arg_i)
        arg_l = np.array(arg_j)
        return arg_k, arg_l

    def svm_train(self, kernel_t, gamma_range, const_range):
        self.kernel = kernel_t
        self.gamma = np.logspace(gamma_range[0], gamma_range[1], 50)
        self.const = np.logspace(const_range[0], const_range[1], 50)
        acc_final = np.empty((len(self.gamma), len(self.const)))
        std_final = np.empty((len(self.gamma), len(self.const)))
        roc_final = np.empty((len(self.gamma), len(self.const)))
        f1_final = np.empty((len(self.gamma), len(self.const)))
        for i in range(0, len(self.gamma)):
            t0 = time.time()
            for j in range(0, len(self.const)):
                param = SVC(C = self.const[j], kernel = self.kernel , gamma = self.gamma[i], class_weight = 'balanced')
                kf = StratifiedKFold(n_splits = 5, shuffle = True)
                acc = []
                roc = []
                f1_ = []
                for train_index, test_index in kf.split(self.train_x, self.train_y):
                   X_train, X_test = self.train_x[train_index], self.train_x[test_index]
                   y_train, y_test = self.train_y[train_index], self.train_y[test_index]
                   list_x = X_train.tolist()
                   list_y = list(y_train)
                   param = param.fit(list_x, list_y)
                   list_test_x = X_test.tolist()
                   list_test_y = list(y_test)
                   accy = param.decision_function(list_test_x)
                   y_pred = param.predict(list_test_x)
                   accu = param.score(list_test_x, list_test_y)
                   roc.append(roc_auc_score(list_test_y, accy))
                   acc.append(accu)
                   f1_.append(f1_score(list_test_y, y_pred))
                acc_mean = np.mean(acc)
                acc_std = np.std(acc)
                acc_final[i, j] = acc_mean
                roc_final[i, j] = np.mean(roc)
                std_final[i, j] = acc_std
                f1_final[i, j] = np.mean(f1_)
            t1 = time.time()
            print("Epoch " + str(i) + " complete! Time taken: " + str(t1-t0))
        file1 = open(self.src + "/results_accuracy.csv", 'w')
        print("The maximum accuracy is: ")
        print(np.max(acc_final))
        arg_g, arg_c = self.argmax(np.max(acc_final), acc_final)
        test_counts = np.empty((len(arg_c), 5))
        file1.write("Gamma,C,STD. DEV,ROC,F1\n")
        k = 0
        if(np.shape(arg_c) == (1,)):
            file1.write(str(self.gamma[arg_g]) + "," + str(self.const[arg_c]) + "," + str(std_final[arg_g, arg_c]) + "," + str(roc_final[arg_g, arg_c]) + "," + str(f1_final[arg_g, arg_c]) + "\n")
            test_counts.append([self.gamma[arg_g], self.const[arg_c], std_final[arg_g, arg_c], roc_final[arg_g, arg_c], f1_final[arg_g, arg_c]])
        else:
            for i, j in zip(arg_c, arg_g):
                file1.write(str(self.gamma[i]) + "," + str(self.const[j]) + "," + str(std_final[i, j]) + "," + str(roc_final[i, j]) + "," + str(f1_final[i, j]) + "\n")
                test_counts[k] = np.array([self.gamma[i], self.const[j], std_final[i, j], roc_final[i, j], f1_final[i, j]])
                k = k + 1
        test_counts[test_counts[:, 4].argsort()]
        self.arg_g = test_counts[0, 0]
        self.arg_c = test_counts[0, 1]
        file1.close()
        file1 = open(self.src + "/results_stddev.csv", 'w')
        print("The minimum standard deviation is: ")
        print(np.min(std_final))
        arg_g, arg_c = self.argmax(np.min(std_final), std_final)
        file1.write("Gamma,C,Accuracy,ROC,F1\n")
        if(np.shape(arg_c) == (1,)):
            file1.write(str(self.gamma[arg_g]) + "," + str(self.const[arg_c]) + "," + str(acc_final[arg_g, arg_c]) + "," + str(roc_final[arg_g, arg_c]) + "," + str(f1_final[arg_g, arg_c]) + "\n")
        else:
            for i, j in zip(arg_c, arg_g):
                file1.write(str(self.gamma[i]) + "," + str(self.const[j]) + "," + str(acc_final[i, j]) + "," + str(roc_final[i, j]) + "," + str(f1_final[i, j]) + "\n")
        file1.close()
        file1 = open(self.src + "/results_ROC.csv", 'w')
        print("The maximum ROC is: ")
        print(np.max(roc_final))
        arg_g, arg_c = self.argmax(np.max(roc_final), roc_final)
        file1.write("Gamma,C,Accuracy,STD. DEV,F1\n")
        if(np.shape(arg_c) == (1,)):
            file1.write(str(self.gamma[arg_g]) + "," + str(self.const[arg_c]) + "," + str(acc_final[arg_g, arg_c]) + "," + str(std_final[arg_g, arg_c]) + "," + str(f1_final[arg_g, arg_c]) + "\n")
        else:
            for i, j in zip(arg_c, arg_g):
                file1.write(str(self.gamma[i]) + "," + str(self.const[j]) + "," + str(acc_final[i, j]) + "," + str(std_final[i, j]) + "," + str(f1_final[i, j]) + "\n")
        file1.close()
        file1 = open(self.src + "/results_F1.csv", "w")
        print("The maximum F1 is: ")
        print(np.max(f1_final))
        file1.write("Gamma,C,Accuracy,ROC,STD. DEV\n")
        arg_g, arg_c = self.argmax(np.max(f1_final), f1_final)
        if(np.shape(arg_c) == (1,)):
            file1.write(str(self.gamma[arg_g]) + "," + str(self.const[arg_c]) + "," + str(acc_final[arg_g, arg_c]) + "," + str(roc_final[arg_g, arg_c]) + "," + str(std_final[arg_g, arg_c]) + "\n")
        else:
            for i, j in zip(arg_c, arg_g):
                file1.write(str(self.gamma[i]) + "," + str(self.const[j]) + "," + str(acc_final[i, j]) + "," + str(roc_final[i, j]) + "," + str(std_final[i, j]) + "\n")
        file1.close()

    def svm_predict(self, gamma_, const_):
        print("The best gamma value, giving first preference to accuracy and next to F1, is: " + str(gamma_))
        print("THe best C value, giving first preference to accuracy and next to F1, is: " + str(const_))
        param = SVC(C = const_, kernel = 'rbf', gamma = gamma_, class_weight = 'balanced')
        list_x = self.test_x.tolist()
        list_y = list(self.test_y)
        param = param.fit(self.train_x, self.train_y)
        accy = param.decision_function(list_x)
        accy2 = param.decision_function(self.train_x.tolist())
        y_pred = param.predict(list_x)
        y_pred2 = param.predict(self.train_x.tolist())
        accu2 = param.score(self.train_x.tolist(), list(self.train_y))
        f1_train = f1_score(y_pred2, list(self.train_y))
        roc2 = roc_auc_score(list(self.train_y), np.ravel(accy2))
        print(y_pred2)
        print(f1_train, accu2, roc2)
        print(y_pred)
        accu = param.score(list_x, list_y)
        roc = roc_auc_score(list_y, np.ravel(accy))
        f1_ = f1_score(list_y, y_pred)
        print("The testing accuracy is: ")
        print(accu)
        print("The testing ROC is: ")
        print(roc)
        print("The testing F1 score is: ")
        print(f1_)



