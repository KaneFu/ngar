from pandas import DataFrame,Series
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


y_df2 = pd.read_csv('data/759funds.csv',index_col=0)
fund_names2 = Series(y_df2.columns)
fund_names = fund_names2.unique()

for name in fund_names:
	temp = []
	f = open('all_output/fund_'+name+'_alpha_mean.txt')
	for line in f.readlines():
		temp.append(line)
	temp2 = np.array(temp)
	temp2 = pd.rolling_mean(temp2,6,min_periods=0)
	np.savetxt('all_output/fund_'+name+'_alpha_mean_ma6.txt',temp2,delimiter='\n')

for name in fund_names:
	temp = []
	f = open('all_output/fund_'+name+'_alpha_mean.txt')
	for line in f.readlines():
		temp.append(line)
	temp2 = np.array(temp)
	temp2 = pd.rolling_mean(temp2,12,min_periods=0)
	np.savetxt('all_output/fund_'+name+'_alpha_mean_ma12.txt',temp2,delimiter='\n')