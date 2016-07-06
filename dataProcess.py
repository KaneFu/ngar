#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Fu Liang(Kane) 
# Email: flburnings@163.com
# Created Time: 2016-06-21 17:58:58
#########################################################################

from pandas import DataFrame,Series
import pandas as pd
import numpy as np

res = pd.read_table('equity_funds_ret.txt',sep='\s+',skiprows=[1,2])

funds = list(pd.value_counts(res['wficn']).index)
# funds.remove(100218)
min_date = 20160331
max_date = 20160331

# for fundID in funds:
# 	temp = res[res['wficn']==fundID].values[0,1] #这里索引不太对
# 	if temp < min_date:
# 		min_date = temp

min_date = 19611229
date_index = pd.period_range(start=str(int(min_date)),end=str(int(max_date)),freq='M')

data = DataFrame(None,index = date_index,columns=funds)

for fundID in funds:
	temp = res[res['wficn']==fundID]
	temp = temp.dropna()
	temp_dateIndex = pd.to_datetime(temp['date'],format = '%Y%m%d')
	temp_series = Series(temp['ret'].values,index = temp_dateIndex)
	temp_series = temp_series.to_period(freq='M')
	data[fundID] = temp_series

data.to_csv('fund_return.csv')