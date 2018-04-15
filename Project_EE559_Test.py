
# coding: utf-8

# In[52]:


import pandas as ps
import sklearn as skl
import numpy as np

data_x = ps.read_csv("bank-additional.csv")
data_x['y_class'] = data_x['y'].apply(lambda x:0 if x == 'no' else 1)
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
print(data_x)
print(data_test)

