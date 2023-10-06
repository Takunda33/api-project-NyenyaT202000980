from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np 
import pandas as pd 
import sqlite3
from collections import Counter
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('sneakers_Reviews_Dataset.csv',sep=';')
print(df.head)

df.shape
df.info()
df['rating'].value_counts()
df.isnull().sum()
df.describe()
df = df.drop('timestamp', axis=1)
df.review_text.head()
df['review_text'] = df['review_text'].str.lower()
df.review_text.head()

pattern = r'[^A-Za-z0-9\s]+'
df['review_text'] = df['review_text'].str.replace(pattern, '', regex=True)

import pandas as pd
import nltk
from nltk.corpus import stopwords  

# Define a function to remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()  
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


df['review_text'] = df['review_text'].apply(remove_stopwords)

replace_map = {'review_text': {'comfortable durable': 1, 'highly recommend': 2, 'recommend': 3, 'buy': 4,
                               'uncomfortable': 5, 'okay': 6, 'poor quality': 7, 'love sneakers': 8, 'great quality': 9,
                              'waste money': 10,'comfortable': 11, 'average sneakers': 12, 'falling apart week': 13, 'decent quality': 14}}

df.replace(replace_map, inplace=True)

replace_map = {'rating': {1: 'negative', 2: 'negative', 3: 'neutral', 4: 'positive',
                                  5: 'positive'}}

df.replace(replace_map, inplace=True)

from sklearn.model_selection import train_test_split
#fixing our data in x and y. Here y contains target data and X contains rest all the features.
x= df.drop(columns= 'rating')
y= df.rating

# splitting our dataset into training and testing for this we will use train_test_split library.
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.30, random_state=42)
print('X_train size: {}, X_test size: {}'.format(x_train.shape, x_test.shape))



RF_model= RandomForestClassifier(n_estimators=100, random_state=42)
RF_model.fit(x_train, y_train)

y_pred_RF= RF_model.predict(x_test)
RF_model.score(x_test,y_test)

accuracy = accuracy_score(y_test, y_pred_RF)
print("Accuracy   :", accuracy)
precision = precision_score(y_test, y_pred_RF, average='weighted')
print("Precision  :", precision)
recall = recall_score(y_test, y_pred_RF, average='weighted')
print("Recall     :", recall)
F1_score = f1_score(y_test, y_pred_RF, average='weighted')
print("F1-score   :", F1_score)

import pickle
RF_model= RandomForestClassifier()

with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(RF_model, model_file)