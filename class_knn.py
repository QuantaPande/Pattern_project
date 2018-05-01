import pandas as pd
import numpy as np
import os as os

class knnPredictor():
    def __init__(self, train_data, test_data, k, risk = True):
        self.train_x = pd.read_csv(train_data)
        self.test_x = pd.read_csv(test_data)
        self.k = k
        self.risk = risk
        end = self.train_x.shape[1]
        self.train_y = self.train_x.values[:, end-1]
        self.train_y = [0 if i < 1 else 1 for i in self.train_y]
        self.test_y = self.test_x.values[:, end-1]
        self.test_y = [0 if i < 1 else 1 for i in self.train_y]

    def eucDist(self, vec_1, vec_2):
        dist = 0    
        dist = np.matmul(vec_2, np.transpose(vec_1))
        np.shape(dist)
        dist = np.sqrt(dist)
        return dist

    def get_lambda(self):
        y_pred = []
        k = self.k
        for i in range(self.train_x.shape[0]):
            dist = np.empty([len(self.train_x), 3])
            dist[:, 0] = self.eucDist(self.train_x.values, self.train_x.values[i, :])
            dist[:, 1] = self.train_y
            dist = dist[dist[:, 0].argsort()]
            labels = dist[1:k+1, 1]
            n1 = 0;
            n2 = 0;
            for j in range(k):
                if(labels[j] == 1):
                    n2 = n2 + 1
                elif(labels[j] == 0):
                    n1 = n1 + 1
                else:
                    ValueError("Improper class labels")
            p_S1 = n1/k
            p_S2 = n2/k
            if(p_S1 > p_S2):
                y_pred.append(0)
            else:
                y_pred.append(1)
        n_1 = 0
        n_2 = 0
        error_1 = 0
        error_2 = 0
        for i in range(len(y_pred)):
            if (y_pred[i] != self.test_y[i]):
                if(self.test_y[i] == 0):
                    error_1 = error_1 + 1
                    n_1 = n_1 + 1
                else:
                    error_2 = error_2 + 1
                    n_2 = n_2 + 1
            else:
                if(self.test_y[i] == 0):
                    n_1 = n_1 + 1
                else:
                    n_2 = n_2 + 1
        acc_1 = 1 - (error_1/n_1)
        acc_2 = 1 - (error_2/n_2)
        acc = 1 - ((error_1 + error_2)/(n_1 + n_2))
        l = np.empty((2, 2))
        l[0,0] = acc_1
        l[0,1] = 1 - acc_1
        l[1,0] = 1 - acc_2
        l[1,1] = acc_2
        print(l)
        return l

    def knn_fit(self):
        y_pred = []
        l = self.get_lambda()
        k = self.k
        for i in range(self.test_x.shape[0]):
            dist = np.empty([len(self.train_x), 3])
            dist[:, 0] = self.eucDist(self.train_x.values, self.test_x.values[i, :])
            dist[:, 1] = self.train_y
            dist = dist[dist[:, 0].argsort()]
            labels = dist[0:k, 1]
            n1 = 0;
            n2 = 0;
            for j in range(k):
                if(labels[j] == 1):
                    n2 = n2 + 1
                elif(labels[j] == 0):
                    n1 = n1 + 1
                else:
                    ValueError("Improper class labels")
            p_S1 = n1/k
            p_S2 = n2/k
            if(self.risk == True):
                if((l[1, 0]-l[0,0])*p_S1 > (l[0,1]-l[1,1])*p_S2):
                    y_pred.append(0)
                else:
                    y_pred.append(1)
            else:
                if(p_S1 > p_S2):
                    y_pred.append(0)
                else:
                    y_pred.append(1)
        return y_pred
    
    def knn_score(self):
        y_pred = self.knn_fit()
        error_1 = 0
        error_2 = 0
        n_1 = 0
        n_2 = 0
        for i in range(len(y_pred)):
            if (y_pred[i] != self.test_y[i]):
                if(self.test_y[i] == 0):
                    error_1 = error_1 + 1
                    n_1 = n_1 + 1
                else:
                    error_2 = error_2 + 1
                    n_2 = n_2 + 1
            else:
                if(self.test_y[i] == 0):
                    n_1 = n_1 + 1
                else:
                    n_2 = n_2 + 1
        acc_1 = 1 - (error_1/n_1)
        acc_2 = 1 - (error_2/n_2)
        acc = 1 - ((error_1 + error_2)/(n_1 + n_2))
        return acc, acc_1, acc_2