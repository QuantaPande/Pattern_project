import pandas as ps
import sklearn as skl
import numpy as np
from sklearn.linear_model import LogisticRegression

class preProc():
    def __init__(self, filename):
        self.data_train = ps.read_csv(filename)
        self.data_train['y_class'] = self.data_train['y'].apply(lambda x:0 if x == 'no' else 1)
        #self.label_train = self.data_train['y_class']
        #self.data_train.drop(['y_class'], axis = 1, inplace = True)
        #self.data_train.drop(['y'], axis = 1, inplace = True)
        self.data_train = self.data_train.replace(to_replace = 'unknown', value = 'NaN')
        drop_terms = []
        for i in self.data_train.index:
            if self.data_train.loc[i, 'education'] == 'illiterate' or self.data_train.loc[i, 'default'] == 'yes':
                drop_terms.append(i)
        self.data_train.drop(drop_terms, axis = 0, inplace = True)
        self.data_test = ps.DataFrame(columns = list(self.data_train))
    
    def splitDataset(self, k):
        for i in list(self.data_train.index.values):
            rand = np.random.random_sample()
            if(rand < k):
                self.data_test = self.data_test.append(self.data_train.loc[i, :])
                self.data_train.drop(i, axis = 0, inplace = True)
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
        #self.train_knn = ps.DataFrame(columns = self.unk_category)
        #self.train_knn[self.unk_category] = self.data_train[self.unk_category]
        self.data_train_enc = ps.DataFrame(columns = list(self.data_train))
        self.data_train_enc = self.data_train.loc[:, [item for item in list(self.data_train) if item not in self.unk_category]]
        print("DONE!!!! Dropped features")
        self.data_train_enc = ps.get_dummies(self.data_train_enc, prefix = None, columns = self.k_category)
        print('Encoded known categorical features')
        #self.train_knn_enc = ps.get_dummies(ps.concat([self.fill_known, self.train_knn], axis = 0), prefix = {'contact':'contact', 'month':'month', 'day_of_week':'day_of_week', 'poutcome':'poutcome'}, columns=['contact', 'month', 'day_of_week', 'poutcome'])
        for i in list(self.data_train_enc)[0: -2]:
            self.data_train_enc.loc[:, i] = (self.data_train_enc.loc[:, i] - np.mean(self.data_train_enc.loc[:, i]))/(np.std(self.data_train_enc.loc[:, i]))
    
    def fit_unknowns(self, data_train, known_index, unknown_index, k, data_train_enc):
        known_index = np.ravel(known_index)
        unknown_index = np.ravel(unknown_index)
        for i in unknown_index:
            #if all([item in self.data_train.columns for item in self.unk_category]):
            #    data_train.drop(self.unk_category, axis = 1, inplace = True)
            knn_loc = self.kNearestNeighbours(data_train_enc.loc[i, :], data_train_enc, k, known_index, self.label_train)
            #data_train[self.unk_category] = train_knn[self.unk_category]
            for j in list(data_train):
                if data_train.loc[i, j] == 'NaN':
                    fitter_value = data_train.loc[knn_loc, j].mode().iloc[0]
                    data_train.loc[i, j] = fitter_value
            #self.train_knn[self.unk_category] = self.data_train[self.unk_category]
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
    
    def processDataset(self, k, split):
        self.splitDataset(split)
        self.idenUnknowns()
        self.oneHotEncoder()
        self.data_train = self.fit_unknowns(self.data_train, self.known_index, self.unknown_index, 5, self.data_train_enc)
        print('Data cleaned! Unknowns filled with best estimates')
        self.data_train.to_csv("bank-additional-train-cleaned.csv", index = False)
        self.numeric = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                        'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
        self.data_train_norm = ps.DataFrame(columns = list(self.data_train))
        self.data_train_norm = self.data_train
        self.data_train_numeric = ps.DataFrame(columns = list(self.numeric))
        self.data_train_numeric[self.numeric] = self.data_train[self.numeric]
        self.data_train_norm[self.numeric] = (self.data_train_numeric-self.data_train_numeric.mean(axis=0, numeric_only=True))/self.data_train_numeric.std(axis=0, numeric_only=True)
        self.data_train_norm = ps.get_dummies(self.data_train_norm, 
                                    prefix = None, 
                                    columns=self.unk_category + self.k_category)
        #self.data_train_norm['y'] = self.label_train['y']
        self.data_train = self.data_train_norm
        #self.data_train = self.dropFeatures(self.data_train)
        self.data_train.drop(['y'], axis = 1, inplace = True)
        self.data_train = ps.concat([self.data_train, self.label_train], axis = 1)
        self.data_train.to_csv("bank-additional-train-cleaned-encoded.csv", index = False)
        self.data_train.drop(['y'], axis = 1, inplace = True)

# =============================================================================
class testProc():
    def __init__(self, pre_data, data_train):
        self.data_train = ps.read_csv(data_train)
        self.data_test = ps.read_csv(pre_data)
        self.rawdata = self.data_test
        self.label_test = ps.DataFrame(columns = ['y'])
        self.label_test = self.rawdata['y']
        self.rawdata.drop(['y'], axis = 1, inplace = True)
         
    def idenUnknowns(self):
        self.train_knn = ps.DataFrame(columns = list(self.rawdata))
        self.fill_known = ps.DataFrame(columns = list(self.rawdata))
        self.known_index = []
        self.unknown_index = []
        for i in self.rawdata.index.tolist():
            if all(self.rawdata.loc[i, :] != 'NaN'):
                self.known_index.append(i)
            else:
                self.unknown_index.append(i)
                self.train_knn = self.train_knn.append(self.rawdata.loc[i, :])
     
    def eucDist(self, vec_1, vec_2):
        dist = 0    
        dist = np.matmul(vec_2, np.transpose(vec_1))
        np.shape(dist)
        dist = np.sqrt(dist)
        return dist
 
    def kNearestNeighbours(self, train_knn, k, data_train):
        dist = np.empty((data_train.shape[0], 2))
        index_i = 0
        for i in data_train.iloc[0]:
            dist[index_i, 0] = self.eucDist(train_knn.values, data_train.loc[i, :].values)
            dist[index_i, 1] = i.astype(int)
            index_i = index_i + 1
        dist = dist[dist[:, 0].argsort()]
        dist_knn = list(dist[1:k+1, 1])
        return dist_knn
         
    def fit_unknowns(self, data_train, unknown_index, train_knn_enc, k):
        unknown_index = np.ravel(unknown_index)
        for i in unknown_index:
            knn_loc = self.kNearestNeighbours(train_knn_enc.loc[i, :], k, data_train_enc)
            for j in list(self.rawdata):
                if self.data_test.loc[i, j] == 'NaN':
                    fitter_value = data_train.loc[knn_loc, j].mode().iloc[0]
                    self.data_test.loc[i, j] = fitter_value
        print(self.data_test.shape)
        return self.data_test
 
    def oneHotEncoder(self):
        print("DONE!!!! Split dataset")
        self.unk_category = ['job', 'marital', 'education', 'default', 'housing', 'loan']
        #self.data_train.drop(['job', 'marital', 'education', 'default', 'housing', 'loan'], axis = 1, inplace = True)
        self.data_train_enc = ps.DataFrame(columns = list(self.data_train))
        self.data_train_enc = self.data_train.loc[:, [item for item in list(self.data_train) if item not in self.unk_category]]
        #self.rawdata.drop(['job', 'marital', 'education', 'default', 'housing', 'loan'], axis = 1, inplace = True)
        self.train_knn.drop(['job', 'marital', 'education', 'default', 'housing', 'loan'], axis = 1, inplace = True)
        print("DONE!!!! Dropped features")
        self.train_knn_enc = ps.get_dummies(self.train_knn, prefix = {'contact':'contact', 'month':'month', 'day_of_week':'day_of_week', 'poutcome':'poutcome'}, columns=['contact', 'month', 'day_of_week', 'poutcome'])
        self.data_train_enc = ps.get_dummies(self.data_train, prefix = {'contact':'contact', 'month':'month', 'day_of_week':'day_of_week', 'poutcome':'poutcome'}, columns=['contact', 'month', 'day_of_week', 'poutcome'])
        for i in list(self.train_knn_enc)[0:-2]:
            self.train_knn_enc.loc[:, i] = (self.train_knn_enc.loc[:, i] - np.mean(self.data_train_enc.loc[:, i]))/(np.std(self.data_train_enc.loc[:, i]))
            self.data_train_enc.loc[:, i] = (self.data_train_enc.loc[:, i] - np.mean(self.data_train_enc.loc[:, i]))/(np.std(self.data_train_enc.loc[:, i]))
     
    def processDataset(self, k):
        self.idenUnknowns()
        self.oneHotEncoder()
        self.rawdata = self.fit_unknowns(self.data_train_enc, self.unknown_index, self.train_knn_enc, 5)
        self.numeric = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                        'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
        self.category = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                    'contact', 'month', 'day_of_week', 'poutcome']
        self.rawdata_numeric = ps.DataFrame(columns = list(self.numeric))
        self.rawdata_numeric[self.numeric] = self.data_test[self.numeric]
        self.rawdata_category = ps.DataFrame(columns = list(self.category))
        self.rawdata_category[self.category] = self.data_test[self.category]
        self.data_train_numeric = ps.DataFrame(columns = list(self.numeric))
        self.data_train_numeric[self.numeric] = self.data_train[self.numeric]
        self.data_train_category = ps.DataFrame(columns = list(self.category))
        self.data_train_category[self.category] = self.data_train[self.category]
        self.rawdata_norm = ps.DataFrame(columns = list(self.rawdata))
        self.rawdata_norm[self.numeric] = (self.rawdata_numeric-self.data_train_numeric.mean(axis=0, numeric_only=True))/self.data_train_numeric.std(axis=0, numeric_only=True)
        self.rawdata_norm[self.category] = self.rawdata_category[self.category]
        #self.rawdata_norm = ps.get_dummies(self.rawdata_norm, prefix = {key: value for (key, value) in zip(self.category, self.category)}, columns = self.category)
        self.rawdata_norm = ps.get_dummies(self.rawdata_norm, 
                                    prefix = {key: value for (key, value) in zip(self.category, self.category)}, 
                                    columns = self.category)
        print(self.rawdata_norm.shape)
        self.data_test = ps.concat([self.rawdata_norm, self.label_test], axis = 1)
        self.data_test.to_csv("bank-additional-test-cleaned-encoded.csv", index = False)
# =================================================================================
class DropFeatures():
    def __init__(self, data_train, data_test):
        self.data_train = ps.read_csv(data_train)
        self.data_test = ps.read_csv(data_test)
        self.label_train = ps.DataFrame(columns = ['y'])
        self.label_train = self.data_train.loc[:, 'y']
        self.label_test = ps.DataFrame(columns = ['y'])
        self.label_test = self.data_test.loc[:, 'y']
        self.data_train.drop(['y'], axis = 1, inplace = True)

    def dropFeatures(self, c):
        log = LogisticRegression(penalty = 'l1', C = c, class_weight = 'balanced', solver = 'saga', max_iter = 100)
        log.fit(self.data_train.values, self.label_train.values)
        weights = log.coef_
        weights = np.ravel(weights)
        inter = log.intercept_
        drop_feature = []
        for i in range(len(weights)):
            if (weights[i] != 0):
                drop_feature.append(list(self.data_train.columns)[i])
        self.data_train_drop = self.data_train.loc[:, drop_feature]
        self.data_test_drop = self.data_test.loc[:, drop_feature]
        self.data_train_drop = ps.concat([self.data_train_drop, self.label_train], axis = 1)
        self.data_test_drop = ps.concat([self.data_test_drop, self.label_test], axis = 1)
        self.data_train_drop.to_csv("bank-additional-train-cleaned-encoded.csv", index = False)
        self.data_test_drop.to_csv("bank-additional-test-cleaned-encoded.csv", index = False)
        print("Features dropped successfully!!!")

class Bootstrap():
    def __init__(self, filename):
        self.data_x = ps.read_csv(filename)
        self.filename = filename

    def bootstrap(self):
        data_x = self.data_x
        data_positive = ps.DataFrame(columns = list(data_x))
        drop_list = []
        for i in list(data_x.index.values):
            if (data_x.loc[i, 'y'] == 1):
                data_positive = data_positive.append(data_x.loc[i, :], ignore_index = True)
                drop_list.append(i)
                data_x.drop([i], axis = 0, inplace = True)
        data_x.reset_index(drop = True)
        N = 2000
        Bootstrap = ps.DataFrame(columns = list(data_x))
        for i in range(0, N):
            r = np.random.randint(0, len(list(data_positive.index.values)))
            Bootstrap = Bootstrap.append(data_positive.loc[r, :], ignore_index = True)
            s = np.random.randint(0, len(list(data_x.index.values)))
            s = data_x.index.values[s]
            Bootstrap = Bootstrap.append(data_x.loc[s, :], ignore_index = True)
        Bootstrap.to_csv(self.filename, index = False)
        