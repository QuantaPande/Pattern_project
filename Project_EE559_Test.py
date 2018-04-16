import pandas as ps
import sklearn as skl
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder

def eucDist(vec_1,vec_2):
    if(len(vec_1) != len(vec_2)):
        raise ValueError("The dimensions of the two vectors must be the same")
    dist = 0    
    for i in range(0, len(vec_1)):
        dist = dist + (vec_1[i] - vec_2[i])**2
    dist = math.sqrt(dist)
    return dist

def kNearestNeighbours(train_knn, data_train, k):
    dist = np.empty((data_train.shape[0], 2))
    index_i = 0
    for i in data_train.index.tolist():
        dist[index_i][0] = eucDist(train_knn.values, data_train.loc[i, :].values)
        dist[index_i][1] = i
        index_i = index_i + 1
    dist = dist[dist[:, 0].argsort()]
    dist_knn = list(dist[1:k+1, 1])
    return dist_knn

    
def fit_unknowns(data_train, known_index, unknwon_index, train_knn_enc, k):
    for i in unknown_index:
        knn_loc = kNearestNeighbours(train_knn_enc.loc[i, :], train_knn_enc.loc[known_index, :], k)
        for j in list(data_train):
            if data_train.loc[i, j] == 'NaN':
                fitter_value = data_train.loc[knn_loc, j].mode().iloc[0]
                data_train.loc[i, j] = fitter_value
    return data_train

    
data_train = ps.read_csv("bank-additional.csv")
data_train['y_class'] = data_train['y'].apply(lambda x:0 if x == 'no' else 1)
label_train = data_train['y_class']
data_train.drop(['y_class'], axis = 1, inplace = True)
data_train.drop(['y'], axis = 1, inplace = True)
data_train = data_train.replace(to_replace = 'unknown', value = 'NaN')
data_test = ps.DataFrame(columns = list(data_train))

# In[53]:

for i in range(0, data_train.shape[0]):
    rand = np.random.random_sample()
    if(rand < 0.1):
        data_test = data_test.append(data_train.loc[i, :])
        data_train.drop(i, axis = 0, inplace = True)
data_train.set_index(np.arange(0, data_train.shape[0]))
data_test.set_index(np.arange(0, data_test.shape[0]))
print(data_train.shape)
print(data_test.shape)

train_knn = ps.DataFrame(columns = list(data_train))
fill_unknown = ps.DataFrame(columns = list(data_train))
known_index = []
unknown_index = []
for i in data_train.index.tolist():
    if all(data_train.loc[i, :] != 'NaN'):
        known_index.append(i)
    else:
        unknown_index.append(i)
    train_knn = train_knn.append(data_train.loc[i, :])

print("DONE!!!! Split dataset")
train_knn.drop(['job', 'marital', 'education', 'default', 'housing', 'loan'], axis = 1, inplace = True)
print("DONE!!!! Dropped features")
train_knn_enc = ps.get_dummies(train_knn, prefix = {'contact':'contacct', 'month':'month', 'day_of_week':'day_of_week', 'poutcome':'poutcome'}, columns=['contact', 'month', 'day_of_week', 'poutcome'])
for i in list(train_knn_enc):
    train_knn_enc.loc[:, i] = train_knn_enc.loc[:, i] - np.mean(train_knn_enc.loc[:, i])

data_train = fit_unknowns(data_train, known_index, unknown_index, train_knn_enc, 5)
data_train.to_csv("bank-additional-cleaned.csv")
print(data_train)


