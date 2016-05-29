import pandas as pd
from pandas import DataFrame,Series
import numpy as np
from datetime import datetime
# import matplotlib.pyplot as plt
from scipy import stats

# res = pd.read_csv("factor.csv",index_col = 0)
# res.index = pd.to_datetime(res.index,format="%m/%d/%y")
# res = pd.rolling_mean(res,6,min_periods=0)
# res.to_csv("factor_ma6.csv")

# res = pd.read_csv("factor.csv",index_col = 0)
# res.index = pd.to_datetime(res.index,format="%m/%d/%y")
# res = pd.rolling_mean(res,12,min_periods=0)
# res.to_csv("factor_ma12.csv")

res = pd.read_csv("759funds.csv",index_col = 0)
res.index = pd.to_datetime(res.index,format="%m/%d/%y")
res = pd.rolling_mean(res,6,min_periods=0)
res.to_csv("759funds_ma6.csv",date_format='%m/%d/%y')

res = pd.read_csv("759funds.csv",index_col = 0)
res.index = pd.to_datetime(res.index,format="%m/%d/%y")
res = pd.rolling_mean(res,12,min_periods=0)
res.to_csv("759funds_ma12.csv",date_format='%m/%d/%y')

factors = pd.read_table('5_factor_daily.txt',sep ='\s+')
factors.index = pd.to_datetime(factors.index,format='%Y%m%d')
factor_mon = factors.resample('M',how ='mean',kind='period')
RF = factor_mon['1996-01':'2015-12']['RF']
RF = RF/12
funds = pd.read_csv('759funds.csv',index_col=0)
RF.index = funds.index
modi_factor = funds.sub(RF,axis=0)
modi_factor.to_csv('759funds_rf.csv')