import numpy as np
import copy
import scipy.io as sio
import math
import random
import datetime
import matplotlib.pyplot as plt
import copy
import time
from pandas import DataFrame,Series
import cProfile,pstats,StringIO
import pandas as pd
from scipy import stats

res  = pd.read_excel('raw_data.xlsx',header = None)
grouped = res.groupby(res[0])

y_df = DataFrame(None)
counter = 1
for name,group in grouped:
	if(len(group)<=60):
		pass
	else:
		# counter=counter+3
		# print group
		temp = DataFrame(group[2])
		temp.index= group[1]
		temp.columns = [name]
		temp.index = pd.to_datetime(temp.index,format='%Y%m%d')
		# print temp
		y_df = pd.merge(y_df,temp,left_index = True,right_index = True,how='outer')


y_df.to_csv(str(y_df.shape[1])+'funds.csv')

