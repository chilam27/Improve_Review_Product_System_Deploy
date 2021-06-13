# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 10:16:48 2021

@author: Chi Lam
"""

import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle

# Load in dataset
df = pd.read_csv('deploy_data.csv')

# Setup
st.write("""
# Predicting Amazon Product's Star Rating Based on Review Comments.

## Product: [Dickies Men's Original 874 Work Pant](https://www.amazon.com/Dickies-Mens-Original-874-Work/dp/B07PFLQ73D/ref=cm_cr_arp_d_product_top?ie=UTF8).
""")

image = Image.open('image.png')
st.image(image, caption = "Product's Image")

# User Input
st.write("""
## Customer Review:
""")

def user_input_features():
    # Review's Date
    from datetime import date
    today = date.today()
    date = today.strftime("%Y")

    # Input Text
    user_input_head = st.text_input("Product Review Header:", "Great value!")
    user_input_cmt = st.text_input("Product Review Comment:", "It has a perfect fit and it is very durable!")
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
    data = {'review_date': date,
            'review_cleaned': user_input,
            'predict_sentiment': sentiment_pred,
            'character_len': character_len,
            'word_count': word_count,
            }
    features = pd.DataFrame(data, index = [len(df)])
    return features

df_input = user_input_features()

# Prepare Data
## Vectorization: Term Frequency-Inverse Document Frequency model (TFIDF) vectorizer + ngrams: bi-gram
ngram_cv = TfidfVectorizer(ngram_range = (2,2))
X = ngram_cv.fit_transform(df_input.review_cleaned)
df_x = pd.DataFrame(X.toarray(), columns = ngram_cv.get_feature_names())

df_input = pd.concat([df_input, df_x], axis = 0)
df_input.iloc[0,5:] = df_input.iloc[1,5:]
df_input.drop(0, inplace = True)

df = pd.concat([df, df_input], axis = 0)
df = df.iloc[:,:39108]
df = df.fillna(0)

## Normalization
norm = MinMaxScaler().fit(df)
user_input = norm.transform(df)[-1]

# Reads in Saved Model
load_log_reg = pickle.load(open('product_review.pkl', 'rb'))

# Make Predictions
input_predict = load_log_reg.predict(user_input.reshape(1,-1))[0]

# Main Panel
st.header('Star Rating Prediction:')
st.write(input_predict, '/5.0')
st.write('---')

# Application Information
st.write("""
## Project Information:

With the idea of reducing customer friction on an e-commerce website, such as Amazon, I built a machine learning that predicts the star rating system (from 1 to 5) based on a review about a particular product. With this implementation, it will improve the traditional product review structure and lead to a better customer experience when leaving a review for a product online.

When a customer wants to leave a review of a product he recently bought, he will need to go through a traditional way of posting a review: write a short header, write down comments, and click on the corresponding star rating for that product. We can do it better by reducing one of the steps that the customer has to go through to complete the process. By implementing my idea of letting the machine predicts the star rating based on the header and comment, the customer saves an extra click of a button. The result of saving an extra click might smoothen the customer' experience, **hence increase the chance of them leaving a review next time**. As Neil Patel also agrees in [one of his blogs](https://neilpatel.com/blog/convince-your-customers-to-review-your-products/) that making it easy for customers to leave a review will have a higher chance of them doing so in the future. **With many more reviews of a product, it will lead to a better conversion rate**. [A study](https://searchengineland.com/88-consumers-trust-online-reviews-much-personal-recommendations-195803) by BrightLocal states that "72% Of Consumers Say That Positive Reviews Make Them Trust A Local Business More."

I was able to build this project using text sentimental analysis and Natural Language Processing (NLP). I created a Log Regression model to predict the rating of the product based on the customer's comment. With the data collected and the Log Regression model, the predictions have an accuracy score being almost *64%*.

To view the project in-depth, check out my ["Improve_Product_System"](https://github.com/chilam27/Improve_Product_Review_System) repository on GitHub where I included a clear ReadMe document of the steps I took from data collection to models evaluation.

To see how I deploy the machine learning model onto the internet using [Heroku](https://www.heroku.com/), check out my ["Improve_Review_System_Deploy"](https://github.com/chilam27/Improve_Review_Product_System_Deploy) repository for the code and data files.
""")
st.write('---')

st.write("""
## Author:

### Dao Chi Lam - [LinkedIn](https://www.linkedin.com/in/chi-dao-lam-0263891a0/), [GitHub](https://github.com/chilam27)
""")

st.write('---')

st.write("""
## Acknowledgments:

[10 Tips to Convince Your Customers to Review Your Products and Generate Social Proof](https://neilpatel.com/blog/convince-your-customers-to-review-your-products/)

[88% Of Consumers Trust Online Reviews As Much As Personal Recommendations](https://searchengineland.com/88-consumers-trust-online-reviews-much-personal-recommendations-195803)
""")
