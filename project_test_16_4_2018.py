import pandas as ps
import sklearn as skl
import numpy as np
import math
import class_preproc

data_x = class_preproc.preProc("bank-additional.csv")
data_x.processDataset(5)
