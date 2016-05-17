#!/usr/bin/env python
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

def prob(alpha_matrix,bin_num,shift_num):
    #alpha_matrix is p*T
    prob_count = np.zeros((bin_num,bin_num))
    prob_matrix = np.zeros((bin_num,bin_num))
    rank_series = alpha_matrix.apply(lambda x:pd.qcut(x,bin_num))
    rank_matrix = np.array([rank_series[i].labels for i in xrange(len(rank_series))])
    
    for ii in range(len(rank_matrix)-shift_num):
        for jj in range(len(rank_matrix[0])):
            if(rank_matrix[ii][jj]!=-1) and (rank_matrix[ii+shift_num][jj]!=-1):
                prob_count[rank_matrix[ii][jj],rank_matrix[ii+shift_num][jj]]+=1

    for ii in range(len(prob_matrix)):
        for jj in range(len(prob_matrix[0])):
            prob_matrix[ii][jj]=prob_count[ii][jj]/(sum(prob_count[ii])+0.0)
    return prob_matrix

res = pd.read_csv('data/759funds.csv',index_col=0)
fund_names = res.columns
ind = res.index

all_alpha = DataFrame(np.zeros((len(ind),len(fund_names))))
for ii in range(len(ind)):
    for jj in range(len(fund_names)):
        all_alpha.ix[ii,jj]=np.nan

for (ii,name) in enumerate(fund_names):
    alpha=np.loadtxt(('all_output/'+'fund_'+name+'_alpha_median.txt'))
    all_alpha.ix[0:(len(alpha)-1),ii]=alpha
print all_alpha
all_alpha = all_alpha[:239]
fund_time = pd.date_range(start = '1996-01-31',end = '2015-11-30',freq='M')
all_alpha.columns = fund_names
all_alpha.index = fund_time
alphas = all_alpha.T
print 'temp'
print alphas
print alphas.shape
cmap = plt.cm.cool
shift_month = 36 
prob_matrix = prob(alphas,5,shift_month)
plt.figure()
plt.imshow(prob_matrix,cmap = cmap)
plt.colorbar()
plt.show()
