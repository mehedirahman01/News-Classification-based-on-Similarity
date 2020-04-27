# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 01:16:14 2020

@author: Mehedi
"""
import pandas as pd
import numpy as np




np.random.seed(500)

df=pd.read_json("Dataset.json",lines=True)


df['category_merged']=df['category'].replace({"HEALTHY LIVING":"WELLNESS","QUEER VOICES": "GROUPS VOICES",
"BUSINESS": "BUSINESS & FINANCES",
"PARENTS": "PARENTING",
"BLACK VOICES": "GROUPS VOICES",
"THE WORLDPOST": "WORLD NEWS",
"STYLE": "STYLE & BEAUTY",
"GREEN": "ENVIRONMENT",
"TASTE": "FOOD & DRINK",
"WORLDPOST": "WORLD NEWS",
"SCIENCE": "SCIENCE & TECH",
"TECH": "SCIENCE & TECH",
"MONEY": "BUSINESS & FINANCES",
"ARTS": "ARTS & CULTURE",
"COLLEGE": "EDUCATION",
"LATINO VOICES": "GROUPS VOICES",
"CULTURE & ARTS": "ARTS & CULTURE",
"FIFTY": "MISCELLANEOUS",
"GOOD NEWS": "MISCELLANEOUS"})


df1=df.drop(['authors','link','date','category'],axis=1)

df1['text']= df1['headline']+" "+df1['short_description']
                                     
df2=df1.drop(['headline','short_description'],axis=1)    
                                 
                                     
                                     
                                     

