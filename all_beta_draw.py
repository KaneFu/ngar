from pandas import DataFrame,Series
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

y_df2 = pd.read_csv('data/759funds.csv',index_col=0)
fund_names2 = Series(y_df2.columns)
fund_names = fund_names2.unique()
ind = y_df2.index


# funds = {}
# for name in fund_names:
# 	funds[name] = []
# 	f = open('all_output/fund_'+name+'_beta.txt')
# 	cols=-1
# 	for line in f.readlines():
# 		if line[0]=='b':
# 			funds[name].append([])
# 			cols=cols+1
# 		else:
# 			funds[name][cols].append(float(line))
# 	f.close()

alphas = {}
for name in fund_names:
	alphas[name]=[]
	f = open('all_output/fund_'+name+'_alpha_median_ma6.txt')
	for line in f.readlines():
		alphas[name].append(float(line))
	f.close()
# print funds

beta_names = ['beta_'+str(num) for num in range(1,5)]
linspace = [(i*100,i*100+99) for i in range(6)]
linspace.append((600,len(fund_names)-1))
counter=1
# for ii in range(4):
	
# 	df = DataFrame(np.zeros((len(ind),len(fund_names))))
# 	df.ix[:,:]=np.NaN
# 	df.index=ind
# 	df.columns=fund_names
# 	for name in fund_names:
# 		for jj in range(len(funds[name][ii])):
# 			df[name][jj]=funds[name][ii][jj]

# 	for area in linspace:
# 		plt.figure(counter)
# 		counter+=1
# 		df.ix[:,area[0]:area[1]].plot(legend=False,ylim=(-0.02,0.02))
# 		png_name = beta_names[ii]+'from_'+str(area[0])+'to_'+str(area[1])+'.png'
# 		plt.savefig('beta_'+str(ii+1)+'/'+png_name,dpi=100)
# 		plt.close()


plt.figure(counter)
df = DataFrame(np.zeros((len(ind),len(fund_names))))
df.ix[:,:]=np.NaN
df.index=ind
df.columns=fund_names
for name in fund_names:
	for jj in range(len(alphas[name])):
		df[name][jj]=alphas[name][jj]

for area in linspace:
	plt.figure(counter)
	counter+=1
	df.ix[:,area[0]:area[1]].plot(legend=False,ylim=(-0.03,0.03))
	png_name = 'alpha'+'from_'+str(area[0])+'to_'+str(area[1])+'ma6.png'
	plt.savefig('alpha/'+png_name,dpi=100)
	plt.close()






