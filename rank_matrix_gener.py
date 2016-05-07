# -*- coding: utf-8 -*-
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

def prob_matrix(alpha_matrix,bin_num,shift_num):
    prob_count_matrix = np.zeros((bin_num,bin_num))
    prob_matrix = np.zeros((bin_num,bin_num))
    rank_series = alpha_matrix.apply(lambda x:pd.qcut(x,bin_num))
    rank_matrix = np.array([rank_series[i].labels for i in xrange(len(rank_series))])

    for ii in xrange(len(rank_matrix)-shift_num):
        for jj in xrange(len(rank_matrix[0])):
            if(rank_matrix[ii,jj]!=-1 and rank_matrix[ii+shift_num,jj]!=-1):
                prob_count_matrix[rank_matrix[ii,jj],rank_matrix[ii+shift_num,jj]] = prob_count_matrix[rank_matrix[ii,jj],rank_matrix[ii+shift_num,jj]]+1
    
    for ii  in xrange(bin_num):
        for jj in xrange(bin_num):
            prob_matrix[ii,jj] = prob_count_matrix[ii,jj]/sum(prob_count_matrix[ii])
    return prob_matrix



# y_df1 = pd.read_csv('data/fund.csv',index_col=0)
# fund_names1 = Series(y_df1.columns)
y_df2 = pd.read_csv('data/759funds.csv',index_col=0)
fund_names2 = Series(y_df2.columns)
fund_names = fund_names2.unique()

alpha_matrix = DataFrame(None)
for ii in range(len(fund_names)):
    try:
        temp_alpha = pd.read_table('all_output/fund_'+fund_names[ii]+'_alpha_median.txt',header =None,names=[fund_names[ii]])
        #print len(temp_alpha)
        # print temp_alpha
        alpha_matrix = pd.merge(alpha_matrix,temp_alpha,how = 'outer',left_index=True,right_index=True)
    except Exception, e:
        pass


	
fund_time = pd.date_range(start='1996-01-31',end='2015-11-30',freq='M')
print("total fund num:", alpha_matrix.shape)

# alpha_matrix.plot(legend = False,ylim=[-0.03,0.03],title=u'所有基金alpha',rot=45)
# alpha_matrix.ix[:,0:30].plot(legend = False,ylim=[-0.03,0.03],title=u'前30支基金alpha',rot=45)
# alpha_matrix.ix[:,30:].plot(legend = False,ylim=[-0.03,0.03],title=u'后40支基金alpha',rot=45)
alpha_matrix.index = fund_time
# alpha_matrix = alpha_matrix.resample('12M',how='mean')
# alpha_matrix = alpha_matrix.groupby(lambda x: x.year).mean()
# alpha_matrix = alpha_matrix.resample('3M',how='mean')
# alpha_matrix = alpha_matrix.T

bin_nums = 5
cmap = plt.cm.cool
counter=1

shift_months = 36
prob = prob_matrix(alpha_matrix, bin_nums, shift_months)
np.savetxt(u'基金三年名次状态转移概率.txt',prob,fmt='%.4f')
plt.figure(counter)
counter+=1
plt.imshow(prob,cmap = cmap)
plt.title(u'3 years freqency')
plt.colorbar()
# plt.show()
plt.savefig(u'基金三年名次状态转移概率.png')
plt.close()

cmap = plt.cm.cool
shift_months = 12
prob = prob_matrix(alpha_matrix, bin_nums, shift_months)
np.savetxt(u'基金年度名次状态转移概率.txt',prob,fmt='%.4f')
plt.figure(counter)
counter+=1
plt.imshow(prob,cmap = cmap)
plt.colorbar()
plt.title(u'anual freqency')
# plt.show()
plt.savefig(u'基金年度名次状态转移概率.png')
plt.close()


shift_months = 6
prob = prob_matrix(alpha_matrix, bin_nums, shift_months)
np.savetxt(u'基金半年名次状态转移概率.txt',prob,fmt='%.4f')
plt.figure(counter)
counter+=1
plt.imshow(prob,cmap = cmap)
plt.title(u'half year freqency')
plt.colorbar()
# plt.show()
plt.savefig(u'基金半年名次状态转移概率.png')
plt.close()

shift_months = 3
prob = prob_matrix(alpha_matrix, bin_nums, shift_months)
np.savetxt(u'基金季度名次状态转移概率.txt',prob,fmt='%.4f')
plt.figure(counter)
counter+=1
plt.imshow(prob,cmap = cmap)
plt.title(u'quarter frequency')
plt.colorbar()
# plt.show()
plt.savefig(u'基金季度名次状态转移概率.png')
plt.close()

shift_months = 1
prob = prob_matrix(alpha_matrix, bin_nums, shift_months)
np.savetxt(u'基金月度名次状态转移概率.txt',prob,fmt='%.4f')
plt.figure(counter)
counter+=1
plt.imshow(prob,cmap = cmap)
plt.title(u'monthly freqency')
plt.colorbar()
# plt.show()
plt.savefig(u'基金月度名次状态转移概率.png')
plt.close()





# shift_months = 36
# prob = prob_matrix(alpha_matrix, bin_nums, shift_months)
# np.savetxt(u'基金三年名次状态转移概率.txt',prob,fmt='%.4f')
# x_list = np.repeat(np.arange(1,bin_nums+1), bin_nums)
# y_list = np.tile(np.arange(1,bin_nums+1), bin_nums)
# plt.scatter(x=x_list, y=y_list,c=prob.reshape(1,bin_nums**2),s=60)
# plt.plot(np.arange(1,11),np.arange(1,11),'k--')
# plt.xlim(0,bin_nums+1)
# plt.ylim(0,bin_nums+1)
# plt.title(u'3 years freqency')
# plt.colorbar()
# plt.show()
# plt.savefig(u'基金三年名次状态转移概率.png')
# plt.close()

# cmap = plt.cm.cool
# shift_months = 12
# prob = prob_matrix(alpha_matrix, bin_nums, shift_months)
# np.savetxt(u'基金年度名次状态转移概率.txt',prob,fmt='%.4f')
# x_list = np.repeat(np.arange(1,bin_nums+1), bin_nums)
# y_list = np.tile(np.arange(1,bin_nums+1), bin_nums)
# # plt.scatter(x=x_list, y=y_list,c=prob.reshape(1,bin_nums**2),s=60)
# plt.imshow(prob,cmap = cmap)
# # plt.plot(np.arange(1,11),np.arange(1,11),'k--')
# plt.colorbar()
# plt.xlim(0,bin_nums+1)
# plt.ylim(0,bin_nums+1)
# plt.title(u'anual freqency')

# plt.show()
# plt.savefig(u'基金年度名次状态转移概率.png')
# plt.close()


# shift_months = 6
# prob = prob_matrix(alpha_matrix, bin_nums, shift_months)
# np.savetxt(u'基金半年名次状态转移概率.txt',prob,fmt='%.4f')
# x_list = np.repeat(np.arange(1,bin_nums+1), bin_nums)
# y_list = np.tile(np.arange(1,bin_nums+1), bin_nums)
# plt.scatter(x=x_list, y=y_list,c=prob.reshape(1,bin_nums**2),s=60)
# plt.plot(np.arange(1,11),np.arange(1,11),'k--')
# plt.xlim(0,bin_nums+1)
# plt.ylim(0,bin_nums+1)
# plt.title(u'half year freqency')
# plt.colorbar()
# plt.show()
# plt.savefig(u'基金半年名次状态转移概率.png')
# plt.close()

# shift_months = 3
# prob = prob_matrix(alpha_matrix, bin_nums, shift_months)
# np.savetxt(u'基金季度名次状态转移概率.txt',prob,fmt='%.4f')
# x_list = np.repeat(np.arange(1,bin_nums+1), bin_nums)
# y_list = np.tile(np.arange(1,bin_nums+1), bin_nums)
# plt.scatter(x=x_list, y=y_list,c=prob.reshape(1,bin_nums**2),s=60)
# plt.plot(np.arange(1,11),np.arange(1,11),'k--')
# plt.xlim(0,bin_nums+1)
# plt.ylim(0,bin_nums+1)
# plt.title(u'quarter frequency')
# plt.colorbar()
# plt.show()
# plt.savefig(u'基金季度名次状态转移概率.png')
# plt.close()

# shift_months = 1
# prob = prob_matrix(alpha_matrix, bin_nums, shift_months)
# np.savetxt(u'基金月度名次状态转移概率.txt',prob,fmt='%.4f')
# x_list = np.repeat(np.arange(1,bin_nums+1), bin_nums)
# y_list = np.tile(np.arange(1,bin_nums+1), bin_nums)
# plt.scatter(x=x_list, y=y_list,c=prob.reshape(1,bin_nums**2),s=60)
# plt.plot(np.arange(1,11),np.arange(1,11),'k--')
# plt.xlim(0,bin_nums+1)
# plt.ylim(0,bin_nums+1)
# plt.title(u'monthly freqency')
# plt.colorbar()
# plt.show()
# plt.savefig(u'基金月度名次状态转移概率.png')
# plt.close()