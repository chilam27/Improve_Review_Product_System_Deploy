# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:37:18 2020

@author: Chi Lam
"""

# Import Library
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load in dataset
df = pd.read_csv('review_cleanned.csv')


# Prepare Data
## Vectorization: Term Frequency-Inverse Document Frequency model (TFIDF) vectorizer + ngrams: bi-gram
ngram_cv = TfidfVectorizer(ngram_range = (2,2))
X = ngram_cv.fit_transform(df.review_cleaned)
df_x = pd.DataFrame(X.toarray(), columns = ngram_cv.get_feature_names())

df = pd.concat([df, df_x], axis = 1)

## Splitting test and train data set
X = df.drop(['rating', 'customer_id', 'customer_name', 'review_header', 'review_body', 'review_txt', 'review_cleaned'], axis = 1)
y = df.rating

X.iloc[:1,:].to_csv('deploy_data.csv', index=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

## Normalization
norm = MinMaxScaler().fit(X_train)
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)

# Model Building: Log Regression
log_reg = LogisticRegression(C = 1, solver = 'lbfgs', multi_class = 'multinomial', random_state = 1).fit(X_train, y_train)

# Saving Model
pickle.dump(log_reg, open('product_review.pkl', 'wb'))
