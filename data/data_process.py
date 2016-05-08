import pandas as pd
from pandas import DataFrame,Series
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats

res = pd.read_csv("factor.csv",index_col = 0)
res.index = pd.to_datetime(res.index,format="%M/%d/%y")
res = pd.rolling_mean(res,6,min_periods=0)
res.to_csv("factor_ma6.csv")

res = pd.read_csv("factor.csv",index_col = 0)
res.index = pd.to_datetime(res.index,format="%M/%d/%y")
res = pd.rolling_mean(res,12,min_periods=0)
res.to_csv("factor_ma12.csv")

res = pd.read_csv("759funds.csv",index_col = 0)
res.index = pd.to_datetime(res.index,format="%M/%d/%y")
res = pd.rolling_mean(res,6,min_periods=0)
res.to_csv("759funds_ma6.csv")

res = pd.read_csv("759funds.csv",index_col = 0)
res.index = pd.to_datetime(res.index,format="%M/%d/%y")
res = pd.rolling_mean(res,12,min_periods=0)
res.to_csv("759funds_ma12.csv")