import pandas as ps
import sklearn as skl
import numpy as np

class preProc():
    def __init__(self, filename):
        self.data_train = ps.read_csv(filename)
        self.data_train['y_class'] = self.data_train['y'].apply(lambda x:0 if x == 'no' else 1)
        #self.label_train = self.data_train['y_class']
        #self.data_train.drop(['y_class'], axis = 1, inplace = True)
        #self.data_train.drop(['y'], axis = 1, inplace = True)
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
        self.label_train = ps.DataFrame(columns = ['y'])
        self.label_train['y'] = self.data_train['y_class']
        self.data_train.drop(['y_class'], axis = 1, inplace = True)
        self.data_train['y'] = self.label_train['y']
        self.data_train.to_csv("bank-additional-train.csv", index = False)
        #self.data_train.drop(['y'], axis = 1, inplace = True)
        self.label_test = ps.DataFrame(columns = ['y'])
        self.label_test['y'] = self.data_test['y_class']
        self.data_test.drop(['y_class'], axis = 1, inplace = True)
        self.data_test['y'] = self.label_test['y']
        self.data_test.to_csv("bank-additional-test.csv", index = False)
        #print(self.data_train.shape)
        #print(self.data_test.shape)
    
    def idenUnknowns(self):
        self.known_index = []
        self.unknown_index = []
        for i in self.data_train.index.tolist():
            if all(self.data_train.loc[i, :] != 'NaN'):
                self.known_index.append(i)
            else:
                self.unknown_index.append(i)
        #print(self.known_index)
    
    def oneHotEncoder(self):
        print("DONE!!!! Split dataset")
        self.unk_category = ['job', 'marital', 'education', 'default', 'housing', 'loan']
        self.k_category = ['contact', 'month', 'day_of_week', 'poutcome']
        self.train_knn = ps.DataFrame(columns = self.unk_category)
        self.train_knn[self.unk_category] = self.data_train[self.unk_category]
        self.data_train.drop(self.unk_category, axis = 1, inplace = True)
        print("DONE!!!! Dropped features")
        self.data_train = ps.get_dummies(self.data_train, prefix = None, columns = self.k_category)
        print('Encoded known categorical features')
        #self.train_knn_enc = ps.get_dummies(ps.concat([self.fill_known, self.train_knn], axis = 0), prefix = {'contact':'contacct', 'month':'month', 'day_of_week':'day_of_week', 'poutcome':'poutcome'}, columns=['contact', 'month', 'day_of_week', 'poutcome'])
        #for i in list(self.train_knn_enc):
        #    self.train_knn_enc.loc[:, i] = self.train_knn_enc.loc[:, i] - np.mean(self.train_knn_enc.loc[:, i])
    
    def fit_unknowns(self, data_train, known_index, unknown_index, k, train_knn):
        known_index = np.ravel(known_index)
        unknown_index = np.ravel(unknown_index)
        for i in unknown_index:
            if all([item in self.data_train.columns for item in self.unk_category]):
                data_train.drop(self.unk_category, axis = 1, inplace = True)
            knn_loc = self.kNearestNeighbours(data_train.loc[i, :], data_train, k, known_index, self.label_train)
            data_train[self.unk_category] = train_knn[self.unk_category]
            for j in list(data_train):
                if data_train.loc[i, j] == 'NaN':
                    fitter_value = data_train.loc[knn_loc, j].mode().iloc[0]
                    data_train.loc[i, j] = fitter_value
            self.train_knn[self.unk_category] = self.data_train[self.unk_category]
        return data_train

    def eucDist(self, vec_1, vec_2):
        dist = 0    
        dist = np.matmul(vec_2, np.transpose(vec_1))
        np.shape(dist)
        dist = np.sqrt(dist)
        return dist

    def kNearestNeighbours(self, train_knn, data_train, k, known_index, label_train):
        dist = np.empty([len(known_index), 3])
        n_cols = list(data_train)
        dist[:, 0] = self.eucDist(train_knn[n_cols[0:-2]].values, data_train.loc[known_index, n_cols[0:-2]].values)
        dist[:, 1] = known_index
        dist[:, 2] = data_train.loc[known_index, 'y']
        dist = dist[dist[:, 0].argsort()]
        dist_knn = np.empty(k)
        label_unk = train_knn.loc['y']
        j = 0
        i = 0
        while (i != 5 and j != data_train.shape[0]):
            if(dist[j, 2] == label_unk):
                dist_knn[i] = dist[j, 1]
                i = i + 1
                j = j + 1
            else:
                j = j + 1
        return dist_knn
        
    def processDataset(self, k):
        self.splitDataset()
        self.idenUnknowns()
        self.oneHotEncoder()
        self.data_train = self.fit_unknowns(self.data_train, self.known_index, self.unknown_index, 5, self.train_knn)
        print(self.data_train)
        self.numeric = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                        'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
        self.data_train_norm = ps.DataFrame(columns = list(self.data_train))
        self.data_train_norm = self.data_train
        
        self.data_train_numeric = ps.DataFrame(columns = list(self.numeric))
        self.data_train_numeric[self.numeric] = self.data_train[self.numeric]
        self.data_train_norm[self.numeric] = (self.data_train_numeric-self.data_train_numeric.mean(axis=0, numeric_only=True))/self.data_train_numeric.std(axis=0, numeric_only=True)

        self.data_train_norm = ps.get_dummies(self.data_train_norm, 
                                    prefix = None, 
                                    columns=self.unk_category)
        self.data_train_norm['y'] = self.label_train['y']
        self.data_train = self.data_train_norm
        #self.data_train = ps.concat([self.data_train, self.label_train], axis = 1)
        self.data_train.to_csv("bank-additional-train-cleaned.csv", index = False)
        #self.data_test = ps.concat([self.data_test, self.label_test], axis = 1)
        self.data_train.drop(['y'], axis = 1, inplace = True)

# =============================================================================
# class testProc():
#     def __init__(self, pre_data, data_train):
#         self.data_test = ps.DataFrame(columns = list(pre_data))
#         self.rawdata = ps.DataFrame(columns = list(pre_data))
#         self.rawdata = pre_data
#         
#     def idenUnknowns(self):
#         self.train_knn = ps.DataFrame(columns = list(self.rawdata))
#         self.fill_known = ps.DataFrame(columns = list(self.rawdata))
#         self.known_index = []
#         self.unknown_index = []
#         for i in self.rawdata.index.tolist():
#             if all(self.rawdata.loc[i, :] != 'NaN'):
#                 self.known_index.append(i)
#                 self.fill_known = self.fill_known.append(self.raw_data.loc[i,:])
#             else:
#                 self.unknown_index.append(i)
#                 self.train_knn = self.train_knn.append(self.raw_data.loc[i, :])
#         print(self.known_index)
#     
#     def eucDist(self, vec_1, vec_2):
#         if(len(vec_1) != len(vec_2)):
#            raise ValueError("The dimensions of the two vectors must be the same")
#         dist = 0    
#         for i in range(0, len(vec_1)):
#             dist = dist + (vec_1[i] - vec_2[i])**2
#         dist = math.sqrt(dist)
#         return dist
# 
#     def kNearestNeighbours(self, train_knn, k, data_train):
#         dist = np.empty((data_train.shape[0], 2))
#         index_i = 0
#         for i in data_train.iloc[0]:
#             dist[index_i, 0] = self.eucDist(train_knn.values, data_train.loc[i, :].values)
#             dist[index_i, 1] = i.astype(int)
#             index_i = index_i + 1
#         dist = dist[dist[:, 0].argsort()]
#         dist_knn = list(dist[1:k+1, 1])
#         return dist_knn
#         
#     def fit_unknowns(self, data_train, known_index, unknown_index, train_knn_enc, k):
#         known_index = np.ravel(known_index)
#         unknown_index = np.ravel(unknown_index)
#         for i in unknown_index:
#             knn_loc = self.kNearestNeighbours(train_knn_enc.loc[i, :], k, data_train_enc)
#             for j in list(self.rawdata):
#                 if self.rawdata.loc[i, j] == 'NaN':
#                     fitter_value = data_train.loc[knn_loc, j].mode().iloc[0]
#                     data_train.loc[i, j] = fitter_value
#         return data_train
# 
#     def oneHotEncoder(self):
#         print("DONE!!!! Split dataset")
#         self.data_train.drop(['job', 'marital', 'education', 'default', 'housing', 'loan'], axis = 1, inplace = True)
#         self.train_knn.drop(['job', 'marital', 'education', 'default', 'housing', 'loan'], axis = 1, inplace = True)
#         print("DONE!!!! Dropped features")
#         self.train_knn_enc = ps.get_dummies(ps.concat([self.fill_known, self.train_knn], axis = 0), prefix = {'contact':'contacct', 'month':'month', 'day_of_week':'day_of_week', 'poutcome':'poutcome'}, columns=['contact', 'month', 'day_of_week', 'poutcome'])
#         self.data_train_enc = ps.get_dummies(self.data_train, prefix = None, columns=['contact', 'month', 'day_of_week', 'poutcome'])
#         for i in list(self.train_knn_enc):
#             self.train_knn_enc.loc[:, i] = self.train_knn_enc.loc[:, i] - np.mean(self.train_knn_enc.loc[:, i])
#     
#     def processDataset(self, k):
#         self.idenUnknowns()
#         self.oneHotEncoder()
#         self.rawdata = self.fit_unknowns(self.rawdata, self.known_index, self.unknown_index, self.train_knn_enc, 5)
#         self.numeric = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
#                         'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
#         self.category = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
#                     'contact', 'month', 'day_of_week', 'poutcome']
#         self.rawdata_numeric = ps.DataFrame(columns = list(self.numeric))
#         self.rawdata_numeric[self.numeric] = self.rawdata[self.numeric]
#         self.rawdata_category = ps.DataFrame(columns = list(self.category))
#         self.rawdata_category[self.category] = self.rawdata[self.category]
#         self.data_train_numeric = ps.DataFrame(columns = list(self.numeric))
#         self.data_train_numeric[self.numeric] = self.data_train[self.numeric]
#         self.data_train_category = ps.DataFrame(columns = list(self.category))
#         self.data_train_category[self.category] = self.data_train[self.category]
#         self.rawdata_norm = ps.DataFrame(columns = list(self.rawdata))
#         self.rawdata_norm[self.numeric] = (self.rawdata_numeric-self.data_train_numeric.mean(axis=0, numeric_only=True))/self.data_train_numeric.std(axis=0, numeric_only=True)
#         self.rawdata_norm[self.category] = self.rawdata_category[self.category]
#         self.rawdata_norm = ps.get_dummies(self.rawdata_norm, 
#                                     prefix = None, 
#                                     columns=self.category)
#         self.data_test = self.rawdata_norm
#         self.data_test.to_csv("bank-additional-test-cleaned.csv", index = False)
# =============================================================================
