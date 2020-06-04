# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:39:26 2020

@author: Mehedi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm,naive_bayes
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle

smote=SMOTE("minority")


df=pd.read_csv("cleaned.csv")
df2=df.sample(frac=1).reset_index(drop=True)


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df2['text_final'],df2['category_merged'],test_size=0.3)


Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)



Tfidf_vect = TfidfVectorizer(max_features=30000)   
Tfidf_vect.fit(df2['text_final'])


Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

X_sm,Y_sm=smote.fit_sample(Train_X_Tfidf,Train_Y)

print(Tfidf_vect.vocabulary_)
print(Train_X_Tfidf)

SVM = svm.SVC(C=1, kernel='linear')
SVM.fit(X_sm,Y_sm)


predictions_SVM = SVM.predict(Test_X_Tfidf)
train_accuracy=SVM.predict(X_sm)

print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
print("Train Accuracy :: ", accuracy_score(Y_sm, train_accuracy))
print(classification_report(Test_Y, predictions_SVM))
results = confusion_matrix(Test_Y,predictions_SVM )
print(results)
##df_confusion = pd.crosstab(Test_Y,predictions_SVM)
##df_confusion.to_csv('your_output_file_name.csv')


##filename = 'finalized_model.sav'
##pickle.dump(SVM, open(filename, 'wb'))

