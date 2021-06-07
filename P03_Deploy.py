# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:37:18 2020

@author: Chi Lam
"""

import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load in dataset
sample_size = 2000
df = pd.read_csv('review_cleanned.csv')
df, _ = train_test_split(df, train_size = sample_size, stratify = df[['rating']], random_state = 1) # reduce the computational power and prevent computer break down
df.reset_index(drop=True, inplace = True)

# Setup
st.write("""
# Predicting Amazon Product's Star Rating Based on Review Comment

## Product: [Dickies Men's Original 874 Work Pant](https://www.amazon.com/Dickies-Mens-Original-874-Work/dp/B07PFLQ73D/ref=cm_cr_arp_d_product_top?ie=UTF8).

Improve the traditional product review system through text sentimental analysis and Natural Language Processing (NLP). By taking only the customer review comment about the product as input, I created a Log Regression model to rate the product based on the customer's comment.
""")

st.write("The sample size for the training model is:", sample_size*0.8)

image = Image.open('image.png')
st.image(image, caption = "Product's Image")

st.write('---')

# User Input
def user_input_features():
    # Review's Date
    from datetime import date
    today = date.today()
    date = today.strftime("%Y")

    # Input Text
    user_input_head = st.text_input("Product Review Header:", '')
    user_input_cmt = st.text_input("Product Review Comment:", '')
    user_input_original = user_input_head + ' ' + user_input_cmt
    
    # Input Cleaning
    def get_wordnet_pos(tag):
        if tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def decontraction(text_string):
        text_string = re.sub(r"n\'t", " not", text_string)
        text_string = re.sub(r"\'re", " are", text_string)
        text_string = re.sub(r"\'s", " is", text_string)
        text_string = re.sub(r"\'d", " would", text_string)
        text_string = re.sub(r"\'ll", " will", text_string)
        text_string = re.sub(r"\'t", " not", text_string)
        text_string = re.sub(r"\'ve", " have", text_string)
        text_string = re.sub(r"\'m", " am", text_string)
        return text_string
    
    def clean_text(text):
        text = text.lower() # convert text to lowercase
        
        text = decontraction(text) # replace contractions with their longer forms
        
        text = re.sub(r'[-();:.,?!"[0-9]+','', text) # remove punctuations and numbers
        
        text = word_tokenize(text) # tokenization
        
        stop_word = stopwords.words('english')
        text = [x for x in text if x not in stop_word] # remove stop words
        
        pos_tags = pos_tag(text)
        text = [WordNetLemmatizer().lemmatize(x[0], get_wordnet_pos(x[1])) for x in pos_tags] # lemmatization
        
        text = [x for x in text if len(x) > 1] # only get word that has more than one character
        
        return text
    
    user_input = " ".join(clean_text(user_input_original))
    if user_input == 0:
        user_input = ''
        
    # Sentiment
    sentiment = SentimentIntensityAnalyzer()

    def sentiment_analysis(compound):
        if compound >= 0.6:
            return '5'
        elif 0.6 > compound >= 0.2:
            return '4'
        elif 0.2 > compound >= -0.2:
            return '3'
        elif -0.2 > compound >= -0.4:
            return '2'
        else:
            return '1'
    
    sentiment_pred = sentiment_analysis(sentiment.polarity_scores(user_input)['compound'])
    
    # Character Length and Word Count
    character_len = len(user_input)
    word_count = len(user_input.split(' '))
    
    # Dataframe
    data = {'customer_id': 'na',
            'customer_name': 'na',
            'rating': 0,
            'review_date': date,
            'review_header': 'na',
            'review_body': 'na',
            'review_txt': 'na',
            'review_cleaned': user_input,
            'predict_sentiment': sentiment_pred,
            'character_len': character_len,
            'word_count': word_count,
            }
    features = pd.DataFrame(data, index = [len(df)])
    return features

df_input = user_input_features()
df = pd.concat([df, df_input], axis = 0)

st.write('---')

# Prepare Data
## Vectorization: Term Frequency-Inverse Document Frequency model (TFIDF) vectorizer + ngrams: bi-gram
ngram_cv = TfidfVectorizer(ngram_range = (2,2))
X = ngram_cv.fit_transform(df.review_cleaned)
df_x = pd.DataFrame(X.toarray(), columns = ngram_cv.get_feature_names())

df = pd.concat([df, df_x], axis = 1)

## Splitting test and train data set

X = df.drop(['rating', 'customer_id', 'customer_name', 'review_header', 'review_body', 'review_txt', 'review_cleaned'], axis = 1)
y = df.rating[:-1]

user_input = X.iloc[-1,:]
X.drop(X.tail(1).index, inplace = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

X_test = X_test.append(user_input, ignore_index = True)

## Normalization
norm = MinMaxScaler().fit(X_train)
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)

user_input = X_test[-1]
X_test = X_test[:-1]

# Model Building: Log Regression
log_reg = LogisticRegression(C = 1, solver = 'lbfgs', multi_class = 'multinomial', random_state = 1).fit(X_train, y_train)

input_predict = log_reg.predict(user_input.reshape(1,-1))[0]

# Main Panel
log_reg_test = log_reg.predict(X_test)
st.header('Star Rating Prediction')
st.write(input_predict, '/5.0', "*(with ", accuracy_score(y_test, log_reg_test), "test accuracy score).*")

