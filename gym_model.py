# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:13:30 2021

@author: hp
"""

import pandas as pd
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

gym=pd.read_excel(r'C:\Users\hp\Downloads\dataGYM.xlsx')

gym.head()

#still 100% accuracy 
X=gym.iloc[:,:3]
X.head()

y=gym.iloc[:,5:]
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.2, random_state=0)

model_gym= RandomForestClassifier(n_estimators=20)

model_gym.fit(X_train, y_train)

expected=y_test
predicted=model_gym.predict(X_test)

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

import pickle

pickle.dump(model_gym, open("model_gym.pkl", "wb"))

model = pickle.load(open("model_gym.pkl", "rb"))

print(model.predict([[40,5.6,70]]))


