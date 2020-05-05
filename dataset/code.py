# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 01:16:14 2020

@author: Mehedi
"""
import pandas as pd
import numpy as np
import matplotlib as plt
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn




np.random.seed(500)

#df=pd.read_json("Dataset.json",lines=True)


#df['category_merged']=df['category'].replace({"HEALTHY LIVING":"WELLNESS","QUEER VOICES": "GROUPS VOICES",
#"BUSINESS": "BUSINESS & FINANCES",
#"PARENTS": "PARENTING",
#"BLACK VOICES": "GROUPS VOICES",
#"THE WORLDPOST": "WORLD NEWS",
#"STYLE": "STYLE & BEAUTY",
#"GREEN": "ENVIRONMENT",
#"TASTE": "FOOD & DRINK",
#"WORLDPOST": "WORLD NEWS",
#"SCIENCE": "SCIENCE & TECH",
#"TECH": "SCIENCE & TECH",
#"MONEY": "BUSINESS & FINANCES",
#"ARTS": "ARTS & CULTURE",
#"COLLEGE": "EDUCATION",
#"LATINO VOICES": "GROUPS VOICES",
#"CULTURE & ARTS": "ARTS & CULTURE",
#"FIFTY": "MISCELLANEOUS",
#"GOOD NEWS": "MISCELLANEOUS"})


#df1=df.drop(['authors','link','date','category'],axis=1)

#df1['text']= df1['headline']+" "+df1['short_description']

#df2=df1.drop(['headline','short_description'],axis=1)

#df2.to_csv (r'merged.csv', index = False, header=True)

df3=pd.read_csv("merged.csv")

#fig=df3['category_merged'].value_counts().plot('bar')

#figure=fig.get_figure()

#figure.savefig('myfile.png', bbox_inches = "tight", dpi=500)

df3['text'] = [entry.lower() for entry in df3['text']]

df3['text']= [word_tokenize(entry) for entry in df3['text']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV



for index,entry in enumerate(df3['text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
       if word not in stopwords.words('english') and word.isalpha():
           word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
           Final_words.append(word_Final)
           
    # The final processed set of words for each iteration will be stored in 'text_final'
    df3.loc[index,'text_final'] = str(Final_words)
    #print(Final_words)

df3.to_csv (r'cleaned.csv', index = False, header=True)