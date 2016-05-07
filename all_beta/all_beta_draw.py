from pandas import DataFrame,Series
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

numbers = ['166','167','183','236']
fund_names = ['fund_'+ num for num in numbers]
funds = {}

for name in fund_names:
	funds[name] = []
	filename = name+'.txt'
	f = open(filename,'r')
	cols=-1
	for line in f.readlines():
		if line[0]=='#':
			funds[name].append([])
			cols=cols+1
		else:
			funds[name][cols].append(float(line))
	f.close()


for ii in range(7):
	df = np.array([funds[name][ii] for  name in fund_names])
	# df = np.transpose(df)
	# df = DataFrame(df,columns = beta_names)
	plt.figure(ii)
	
	for jj in range(4):
		plt.plot(df[jj],label=fund_names[jj])
	plt.legend()

	png_name = 'beta_'+str(ii)+'.png'
	plt.savefig(png_name,dpi=100)







