#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:15:31 2023

@author: ealgar
"""
import spacy
import pandas as pd
import string
import joblib
import re
import argparse

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# Emotions
from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet

import shap

#%%
nlp = spacy.load("es_core_news_sm")
#%%

parser = argparse.ArgumentParser(description='Emotions and Fakeness Probability')
parser.add_argument('-i', '--input_path', type=str, help='File/dataset (csv) path ex:/home/file.csv')
parser.add_argument('-o', '--output_path', type=str, help='File/dataset (csv) path to save results ex:/home/file_result.csv')
parser.add_argument('-rf', '--rf_path', type=str, help='Random forest path ex:/home/RFmodel.joblib')
parser.add_argument('-c', '--text_col', type=str, help='Name of text column in the csv')

args = parser.parse_args()

# Args values
input_path = args.input_path
output_path = args.output_path
rf_path = args.rf_path
text_col = args.text_col

# print values to verify
print(f'Input Path: {input_path}')
print(f'Output Path: {output_path}')
print(f'Random Forest Path: {rf_path}')
print(f'Text Column Name: {text_col}')

df = pd.read_csv(input_path)
text_hoax = df[text_col]
lang = 'spanish'

#%%

# Emotions to extract
emotion_cols = [
        'not ironic', 
        'ironic', 
        'hateful', 
        'targeted', 
        'aggressive', 
        'others', 
        'joy', 
        'sadness', 
        'anger', 
        'surprise', 
        'disgust', 
        'fear', 
        'NEG', 
        'NEU', 
        'POS', 
        'REAL', 
        'FAKE', 
        'toxic', 
        'very_toxic'
        ]



# Sentiment analysis models
MODELS = [
    create_analyzer(model_name="pysentimiento/robertuito-irony", lang="es"),
    create_analyzer(model_name="pysentimiento/robertuito-hate-speech", lang="es"),
    create_analyzer(model_name="pysentimiento/robertuito-emotion-analysis", lang="es"),
    create_analyzer(model_name="Newtral/xlm-r-finetuned-toxic-political-tweets-es", lang="es"),
    create_analyzer(model_name="pysentimiento/robertuito-sentiment-analysis", lang="es"),
    create_analyzer(model_name="Narrativaai/fake-news-detection-spanish", lang="es")
]
#%%

# Sentiment/emotions extraction
def text_mining(text):
    df_probas = {}
    for idx, classifier in enumerate(MODELS):
        prediction = classifier.predict(text)
        df_probas.update(prediction.probas)
    
    v = df_probas['LABEL_0']
    del df_probas['LABEL_0']
    df_probas['toxic'] = v
    v = df_probas['LABEL_1']
    del df_probas['LABEL_1']
    df_probas['very_toxic'] = v
   
    return df_probas   


# POS TAGS
def count_tags(text):
    doc = nlp(text)
    count_entities = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  #[ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X]
    for token in doc:
        if not any(x in token.text for x in string.punctuation) and not token.text.startswith('https'):
            tag = token.pos_
            if tag == "ADJ":
                count_entities[0] += 1
            elif tag == "ADP":
                count_entities[1] += 1
            elif tag == "ADV":
                count_entities[2] += 1
            elif tag == "AUX":
                count_entities[3] += 1
            elif tag == "CCONJ":
                count_entities[4] += 1
            elif tag == "DET":
                count_entities[5] += 1
            elif tag == "INTJ":
                count_entities[6] += 1
            elif tag == "NOUN":
                count_entities[7] += 1
            elif tag == "NUM":
                count_entities[8] += 1
            elif tag == "PART":
                count_entities[9] += 1
            elif tag == "PRON":
                count_entities[10] += 1
            elif tag == "PROPN":
                count_entities[11] += 1
            elif tag == "PUNCT":
                count_entities[12] += 1
            elif tag == "SCONJ":
                count_entities[13] += 1
            elif tag == "SYM":
                count_entities[14] += 1
            elif tag == "VERB":
                count_entities[15] += 1
            elif tag == "X":
                count_entities[16] += 1    
    return count_entities


    
# Entity types
def extract_entity_types(text):
    doc = nlp(text)
    count_entities = [0,0,0,0,0,0,0, 0] #[PERSON, ORG, GPE-Nombre pais, DATE, CARDINAL, MONEY, PRODUCT, OTROS]
    for x in doc.ents:
        if not x.text.startswith('https'):
            tag = x.label_
            if tag == "PERSON":
                count_entities[0] += 1
            elif tag == "ORG":
                count_entities[1] += 1
            elif tag == "GPE":
                count_entities[2] += 1
            elif tag == "DATE":
                count_entities[3] += 1
            elif tag == "CARDINAL":
                count_entities[4] += 1
            elif tag == "MONEY":
                count_entities[5] += 1 
            elif tag == "PRODUCT":
                count_entities[6] += 1 
            else:
                count_entities[7] += 1
    return count_entities

# Sentences
def count_sentences(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    n_sentences = len(sentences)
    return n_sentences

#TF-IDF
def generate_list_tfidf_values(texts):
    stop_words = set(stopwords.words(lang))

    lemmatizer = WordNetLemmatizer()

    tfidf_values_list = []

    for text in texts:
        words = word_tokenize(text.lower())

        filtered_words = [word for word in words if word not in stop_words]

        frequency = FreqDist(filtered_words)

        words_lematized = [lemmatizer.lemmatize(word) for word in filtered_words] 

        pos_tags = pos_tag(words_lematized)

        frequency_pos = FreqDist(tag for palabra, tag in pos_tags)

        # Calculate TF-IDF for each word
        tfidf = defaultdict(float)
        total_palabras = len(words_lematized)
        for word in words_lematized:
            tf = frequency[word] / total_palabras
            idf = 1 / (1 + frequency_pos[pos_tag([word])[0][1]])
            tfidf[word] = tf * idf

        # Obtain TF-IDF values
        valores = list(tfidf.values())

        tfidf_values_list.append(valores)

    return tfidf_values_list

def get_name(x):
    match = re.search(r'"(name)": "([^"]*)"', x)
    if match:
        return match.group(2)
    return ''

def extract_image_url(extended_user):
    pattern = r'https://pbs.twimg.com/profile_images/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    match = re.search(pattern, extended_user)
    if match:
        return match.group(0)
    else:
        return None

#%%
        
# TF-IDF y POS
tag_list = []
for t in text_hoax:
    tag_list.append(count_tags(t)) 
    
entity_types_list = []
for i in text_hoax:
    entity_types_list.append(extract_entity_types(i))
    
sentence_list = []
for t in text_hoax:
    sentence_list.append(count_sentences(t))

tfidf_values_list = generate_list_tfidf_values(text_hoax)

df['POS_tags'] = tag_list
df['POS_entities'] = entity_types_list
df['sentences'] = sentence_list
df['TFIDF'] = tfidf_values_list

# Check if any row in column 'TFIDF_1d' has less than 2 elements
mask = df['TFIDF'].apply(lambda x: len(x) < 2)

# Delete rows that meet the condition
df = df[~mask]


POS_entities_1d = []
POS_tags_1d = []
TFIDF_1d = []
 
# Dimension reduction
for i in df['POS_tags']:
    X_dense = i
    X = csr_matrix(X_dense)
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=None)
    svd.fit(X)
    POS_tags_1d.append(svd.singular_values_[0])


for i in df['POS_entities']:
    X_dense = i
    X = csr_matrix(X_dense)
    svd = TruncatedSVD(n_components=1, n_iter=5, random_state=None)
    svd.fit(X)
    POS_entities_1d.append(svd.singular_values_[0])


for i in df['TFIDF']:
    X_dense = i
    X = csr_matrix(X_dense)
    svd = TruncatedSVD(n_components=1, n_iter=5, random_state=None)
    svd.fit(X)
    TFIDF_1d.append(svd.singular_values_[0])


df['POS_entities_1d'] = POS_entities_1d
df['POS_tags_1d'] = POS_tags_1d
df['TFIDF_1d'] = TFIDF_1d


# EMOTIONS
# Extraction of emotions and feelings
tweet_encoddings_vec = []
for index, row in df.iterrows():
    print('-', index)
    tweet_encoddings_vec.append(preprocess_tweet(row[text_col]))
    dict_charact = text_mining(row[text_col])
    
    for col in emotion_cols:
        df.at[index, col]  = dict_charact[col]

# RANFOM FOREST
prob_cols = [
    'not ironic', 
    #'ironic', 
    'hateful', 
    'targeted', 
    'aggressive', 
    'others', 
    'joy', 
    'sadness', 
    'anger', 
    'surprise', 
    'disgust', 
    'fear', 
    'NEG', 
    # 'NEU', 
    'POS', 
    'REAL', 
    # 'FAKE', 
    'toxic', 
    'very_toxic',
    'POS_tags_1d',
    'POS_entities_1d',
    'sentences',
    'TFIDF_1d'
]

# Load RF
loaded_forest = joblib.load(rf_path)

# Making predictions 
fakeness = loaded_forest.predict(df[prob_cols])

# Making probability predictions instead of binary labels
fakeness_probabilities = loaded_forest.predict_proba(df[prob_cols])

# Example: obtaining the probabilities of class 1 - fakeness
fakeness_probabilities_class_1 = fakeness_probabilities[:, 1]

# FAKENESS
df['fakeness'] = fakeness
df['fakeness_probabilities'] = fakeness_probabilities_class_1

 
# Rename columns
df.rename(columns={"NEU":"Neutral", "POS":"Positive", "NEG": "Negative"}, inplace=True)

# Sort by date 
df = df.sort_values(by='date')
    
# Save dataset
df.to_csv(output_path)
df.info()
