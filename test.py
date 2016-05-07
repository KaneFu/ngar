#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Fu Liang(Kane) 
# Email: flburnings@163.com
# Created Time: 2016-04-20 13:33:13
#########################################################################

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

beta_num=6
data_len = 250

x_list = np.zeros((beta_num,data_len))
for ii in range(beta_num):
	for jj in range(data_len):
		x_list[ii][jj]=np.random.random()
for jj in range(data_len):
	x_list[0][jj]=1

beta_list = np.zeros((beta_num,data_len))
for jj in range(data_len):
	beta_list[0][jj]=2.3+jj*0.01
	beta_list[1][jj]=2.3+(jj-123)/23.3
	beta_list[2][jj]=2.2+(jj/150.0)**2
	beta_list[3][jj]=2.5-jj*0.01
	beta_list[4][jj]=0.05*np.random.random()+2
	beta_list[5][jj]=0.3+abs(jj/2.0-123)

y_list = np.zeros(data_len)
for jj in range(data_len):
	y_list[jj]=sum([beta_list[p][jj]*x_list[p][jj] for p in range(beta_num)])+0.01*np.random.random()

# for ii in range(beta_num):
# 	plt.figure(ii+1)
# 	plt.plot(beta_list[ii])
# 	plt.title("beta"+str(ii))
# 	plt.xlabel("T")
# 	plt.ylabel("Beta")
# 	png_name ="test/"+"beta"+str(ii)+".png"
# 	plt.savefig(png_name,dpi=100)
# 	plt.close()

x_df = DataFrame(x_list.T)
x_df.to_csv("test/x_list.csv")
y_df = DataFrame(y_list)
y_df.to_csv("test/y_list.csv")
