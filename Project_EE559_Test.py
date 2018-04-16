
# coding: utf-8

# In[52]:


import pandas as ps
import sklearn as skl
import numpy as np
import math

def eucDist(vec_1,vec_2):
    if(length(vec_1) != length(vec_2)):
        raise ValueError("The dimensions of the two vectors must be the same")
        break
    dist = 0    
    for i in range(0, length(vec_1)):
        dist = dist + (vec_1[i] - vec_2[i])**2
    dist = math.sqrt(dist)
    return dist

def kNearestNeighbours(train_knn, k):
    for i in list(train_knn):
        train_knn[:, i] = train_knn[:, i] - np.mean(train_knn[:, i])

    dist = np.empty(shape = (np.shape(train_knn)[0], np.shape(train_knn)[0], 3))
    index_i = 0
    index_j = 0
    for i in train_knn.index.tolist():
        for j in train_knn.index.tolist():
            dist[index_i][index_j][0] = eucDist(train_knn[i, :], train_knn[j, :])
            dist[index_i][index_j][1] = i
            dist[index_i][index_j][2] = j
            index_j = index_j + 1
        index_j = 0
        index_i = index_i + 1

    for i in range(0, np.shape(dist)[0]):
        train_knn[i, :, :] = train_knn[train_knn[i, :, 0].argsort()]

    dist_knn = dist[1:k+1, :, 1:2]
    
def fit_unknowns(data_x, known_index, fill_unknown, knn_loc):
    
    
    
data_x = ps.read_csv("bank-additional.csv")
data_x['y_class'] = data_x['y'].apply(lambda x:0 if x == 'no' else 1)
label_train = data_x['y_class']
data_x.drop(['y_class'], axis = 1, inplace = True)
data_x.drop(['y'], axis = 1, inplace = True)
data_x = data_x.replace(to_replace = 'unknown', value = 'NaN')
data_test = ps.DataFrame(columns = list(data_x))
print(data_x.shape)


# In[53]:


for i in range(0, data_x.shape[0]):
    rand = np.random.random_sample()
    if(rand < 0.1):
        data_test = data_test.append(data_x.loc[i, :])
        data_x.drop(i, axis = 0)
data_x.set_index(np.arange(0, data_x.shape[0]))
data_test.set_index(np.arange(0, data_test.shape[0]))
#print(data_x)
#print(data_test)

knn_loc = kNearestNeighbours(train_knn, 5)

