
# coding: utf-8

# In[25]:


import pandas as ps
import sklearn as skl
import numpy as np
data_x = ps.read_csv("bank-additional.csv")
data_x['y_class'] = data_x['y'].apply(lambda x:0 if x == 'no' else 1)
data_x.drop(['y'], axis = 1, inplace = True)
data_x = data_x.replace(to_replace = 'unknown', value = 'NaN')
data_test = ps.DataFrame(columns = list(data_x))
print(data_test)
print(data_x)

