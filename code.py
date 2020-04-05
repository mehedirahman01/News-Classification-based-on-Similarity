# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 01:16:14 2020

@author: Mehedi
"""
import pandas as pd
import numpy as np




np.random.seed(500)


df= pd.read_csv(r'dataset\Dataset.csv',encoding='latin-1')

x=df.drop(['authors','link','date'],axis=1)


