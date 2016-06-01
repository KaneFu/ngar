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

def series_to_rank(series,bin_num):
    # series is series
    bin_len = alpha_matrix.shape[0]/bin_num
    origin_index = series.index
    df = DataFrame(series)
    df['rank']=1
    df.columns = ['time','rank']
    df=df.sort(columns='time',ascending=False)
    for ii in range(bin_num-1):
        df['rank'][ii*bin_len:(ii+1)*bin_len]=ii+1
    df['rank'][(bin_num-1)*bin_len:]=bin_num
    df['rank'][df['time'].isnull()]=bin_num
    df = df.reindex(origin_index)
    return df['rank'].values

def prob_matrix(alpha_matrix,bin_num,shift_num,func):
    prob_count_matrix = np.zeros((bin_num,bin_num))
    prob_matrix_temp = np.zeros((bin_num,bin_num))
    rank_matrix = alpha_matrix.apply(func,args=([bin_num]))
    rank_matrix = rank_matrix.values
    print rank_matrix.shape

    for ii in xrange(rank_matrix.shape[0]):
        for jj in xrange(rank_matrix.shape[1]-shift_num):
            prob_count_matrix[rank_matrix[ii,jj]-1,rank_matrix[ii,jj+shift_num]-1] = prob_count_matrix[rank_matrix[ii,jj]-1,rank_matrix[ii,jj+shift_num]-1]+1
    
    for ii  in xrange(bin_num):
        for jj in xrange(bin_num):
            prob_matrix_temp[ii,jj] = prob_count_matrix[ii,jj]/sum(prob_count_matrix[ii])
    return prob_matrix_temp




# def prob_matrix(alpha_matrix,bin_num,shift_num):
#     bin_len = alpha_matrix.shape[0]/bin_num
#     prob_count_matrix = np.zeros((bin_num,bin_num))
#     prob_matrix_temp = np.zeros((bin_num,bin_num))


#     for ii  in xrange(bin_num):
#         for jj in xrange(bin_num):
#             prob_matrix_temp[ii,jj] = prob_count_matrix[ii,jj]/sum(prob_count_matrix[ii])
#     return prob_matrix_temp


# y_df1 = pd.read_csv('data/fund.csv',index_col=0)
# fund_names1 = Series(y_df1.columns)
y_df2 = pd.read_csv('data/759funds.csv',index_col=0)
fund_names2 = Series(y_df2.columns)
fund_names = fund_names2.unique()
bin_nums = [5,10]
choose_types = ['median','median_ma6','median_ma12','mean','mean_ma6','mean_ma12']
for bin_num in bin_nums:
    for choose_type in choose_types:
        alpha_matrix = DataFrame(None)
        for ii in range(len(fund_names)):
            try:
                temp_alpha = pd.read_table('all_output/fund_'+fund_names[ii]+'_alpha_'+choose_type+'.txt',header =None,names=[fund_names[ii]])
                #print len(temp_alpha)
                # print temp_alpha
                alpha_matrix = pd.merge(alpha_matrix,temp_alpha,how = 'outer',left_index=True,right_index=True)
            except Exception, e:
                pass
        
        folder = ''
        if bin_num ==5:
            if choose_type[:4]=='medi':
                folder = u'五份median/'
            else:
                folder = u'五份mean/'
        else:
            if choose_type[:4]=='medi':
                folder = u'十份median/'
            else:
                folder = u'十份mean/'


        fund_time = pd.date_range(start='1996-01-31',end='2015-11-30',freq='M')
        print("total fund num:", alpha_matrix.shape)
        
        alpha_matrix.index = fund_time
        alpha_matrix = alpha_matrix.T


        cmap = plt.cm.cool
        counter=1
        
        shift_months = 36
        prob = prob_matrix(alpha_matrix, bin_num, shift_months,series_to_rank)
        np.savetxt(folder+'nums/'+u'三年转移概率'+choose_type+'.txt',prob,fmt='%.4f')
        plt.figure(counter)
        counter+=1
        plt.imshow(prob,cmap = cmap)
        plt.title(u'3 years freqency')
        plt.colorbar()
        # plt.show()
        plt.savefig(folder+u'三年转移概率'+choose_type+'.png')
        plt.close()
        
        cmap = plt.cm.cool
        shift_months = 12
        prob = prob_matrix(alpha_matrix, bin_num, shift_months,series_to_rank)
        np.savetxt(folder+'nums/'+u'年度转移概率'+choose_type+'.txt',prob,fmt='%.4f')
        plt.figure(counter)
        counter+=1
        plt.imshow(prob,cmap = cmap)
        plt.colorbar()
        plt.title(u'anual freqency')
        # plt.show()
        plt.savefig(folder+u'年度转移概率'+choose_type+'.png')
        plt.close()
        
        
        shift_months = 6
        prob = prob_matrix(alpha_matrix, bin_num, shift_months,series_to_rank)
        np.savetxt(folder+'nums/'+u'半年转移概率'+choose_type+'.txt',prob,fmt='%.4f')
        plt.figure(counter)
        counter+=1
        plt.imshow(prob,cmap = cmap)
        plt.title(u'half year freqency')
        plt.colorbar()
        # plt.show()
        plt.savefig(folder+u'半年转移概率'+choose_type+'.png')
        plt.close()
        
        shift_months = 3
        prob = prob_matrix(alpha_matrix, bin_num, shift_months,series_to_rank)
        np.savetxt(folder+'nums/'+u'季度转移概率'+choose_type+'.txt',prob,fmt='%.4f')
        plt.figure(counter)
        counter+=1
        plt.imshow(prob,cmap = cmap)
        plt.title(u'quarter frequency')
        plt.colorbar()
        # plt.show()
        plt.savefig(folder+u'季度转移概率'+choose_type+'.png')
        plt.close()
        
        shift_months = 1
        prob = prob_matrix(alpha_matrix, bin_num, shift_months,series_to_rank)
        np.savetxt(folder+'nums/'+u'月度转移概率'+choose_type+'.txt',prob,fmt='%.4f')
        plt.figure(counter)
        counter+=1
        plt.imshow(prob,cmap = cmap)
        plt.title(u'monthly freqency')
        plt.colorbar()
        # plt.show()
        plt.savefig(folder+u'月度转移概率'+choose_type+'.png')
        plt.close()
        
        
        
        
        
# shift_months = 36
# prob = prob_matrix(a,series_to_ranklpha_matrix, bin_num, shift_months)
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