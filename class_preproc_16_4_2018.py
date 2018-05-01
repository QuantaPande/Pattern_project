import pandas as ps
import sklearn as skl
import numpy as np
import math

class preProc():
    def __init__(self, filename):
        self.data_train = ps.read_csv(filename)
        self.data_train['y_class'] = self.data_train['y'].apply(lambda x:0 if x == 'no' else 1)
        self.label_train = self.data_train['y_class']
        self.data_train.drop(['y_class'], axis = 1, inplace = True)
        self.data_train.drop(['y'], axis = 1, inplace = True)
        self.data_train = self.data_train.replace(to_replace = 'unknown', value = 'NaN')
        self.data_test = ps.DataFrame(columns = list(self.data_train))
    
    def splitDataset(self):
        for i in range(0, self.data_train.shape[0]):
            rand = np.random.random_sample()
            if(rand < 0.1):
                self.data_test = self.data_test.append(self.data_train.loc[i, :])
                self.data_train.drop(i, axis = 0, inplace = True)
        self.data_train.set_index(np.arange(0, self.data_train.shape[0]))
        self.data_test.set_index(np.arange(0, self.data_test.shape[0]))
        print(self.data_train.shape)
        print(self.data_test.shape)
    
    def idenUnknowns(self):
        self.train_knn = ps.DataFrame(columns = list(self.data_train))
        self.fill_unknown = ps.DataFrame(columns = list(self.data_train))
        self.known_index = []
        self.unknown_index = []
        for i in self.data_train.index.tolist():
            if all(self.data_train.loc[i, :] != 'NaN'):
                self.known_index.append(i)
            else:
                self.unknown_index.append(i)
                self.train_knn = self.train_knn.append(self.data_train.loc[i, :])
    
    def eucDist(self, vec_1, vec_2):
        if(len(vec_1) != len(vec_2)):
           raise ValueError("The dimensions of the two vectors must be the same")
        dist = 0    
        for i in range(0, len(vec_1)):
            dist = dist + (vec_1[i] - vec_2[i])**2
        dist = math.sqrt(dist)
        return dist

    def kNearestNeighbours(self, train_knn, data_train, known_index, k):
        dist = np.empty((data_train.shape[0], 2))
        index_i = 0
        for i in known_index:
            dist[index_i][0] = self.eucDist(train_knn.values, data_train.loc[i, :].values)
            dist[index_i][1] = i
            index_i = index_i + 1
        dist = dist[dist[:, 0].argsort()]
        dist_knn = list(dist[1:k+1, 1])
        return dist_knn

        
    def fit_unknowns(self, data_train, known_index, unknwon_index, train_knn_enc, k):
        for i in self.unknown_index:
            knn_loc = self.kNearestNeighbours(train_knn_enc.loc[i, :], train_knn_enc.loc[known_index, :], k)
            for j in list(data_train):
                if data_train.loc[i, j] == 'NaN':
                    fitter_value = data_train.loc[knn_loc, j].mode().iloc[0]
                    data_train.loc[i, j] = fitter_value
        return data_train

    def oneHotEncoder(self):
        print("DONE!!!! Split dataset")
        self.train_knn.drop(['job', 'marital', 'education', 'default', 'housing', 'loan'], axis = 1, inplace = True)
        print("DONE!!!! Dropped features")
        self.train_knn_enc = ps.get_dummies(self.train_knn, prefix = {'contact':'contacct', 'month':'month', 'day_of_week':'day_of_week', 'poutcome':'poutcome'}, columns=['contact', 'month', 'day_of_week', 'poutcome'])
        for i in list(self.train_knn_enc):
            self.train_knn_enc.loc[:, i] = self.train_knn_enc.loc[:, i] - np.mean(self.train_knn_enc.loc[:, i])
    
    def processDataset(self, k):
        self.splitDataset()
        self.idenUnknowns()
        self.oneHotEncoder()
        self.data_train = self.fit_unknowns(self.data_train, self.known_index, self.unknown_index, self.train_knn_enc, 5)
        self.data_train.to_csv("bank-additional-cleaned.csv", index = False)
        print(self.data_train)
        self.numeric = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                        'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
        self.category = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                    'contact', 'month', 'day_of_week', 'poutcome']
        self.data_train_numeric = ps.DataFrame(columns = list(self.numeric))
        self.data_train_numeric[self.numeric] = self.data_train[self.numeric]
        self.data_train_category = ps.DataFrame(columns = list(self.category))
        self.data_train_category[self.category] = self.data_train[self.category]
        self.data_train_norm = (self.data_train_numeric-self.data_train_numeric.mean(axis=0, numeric_only=True))/self.data_train_numeric.std(axis=0, numeric_only=True)
        self.data_train_norm[self.category] = self.data_train_category[self.category]
        self.data_train_norm = ps.get_dummies(self.data_train_norm, 
                                    prefix = None, 
                                    columns=self.category)
        self.data_train = self.data_train_norm.values
        print(self.data_train_norm)
