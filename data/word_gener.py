#!/usr/bin/env python
# coding=utf-8
import pandas as pd
from pandas import DataFrame,Series
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
from docx import Document
import os


root_dic = '/Users/BurNingS/Desktop/prob_trans/'
all_dic = os.listdir(root_dic)[1:]

for dic in all_dic:
	items= os.listdir(root_dic+dic)
	all_txt = [item[:-4] for item in items if item[-3:]=='txt']


	for item in all_txt:
		res = pd.read_table(root_dic+dic+'/'+item+'.txt',sep='\s+',header =None)
		text = str(res.values)
		text = text.replace('[', ' ',1)
		text = text.replace('[', ' ')
		text = text.replace(']', ' ')
		document = Document()
		document.add_paragraph(text=text,style=None)
		document.save(root_dic+dic+'/'+item+'.docx')
	
	