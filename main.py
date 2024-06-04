# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

        
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


import nltk
from nltk.tokenize import word_tokenize
import spacy
import numpy as np
from collections import Counter

nltk.download('punkt')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from wordfreq import word_frequency

from transformers import BertTokenizer, BertModel
import torch

# Load necessary libraries and models
nltk.download('punkt')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Helper functions
def count_consonants(word):
    return len([char for char in word if char.lower() in 'bcdfghjklmnpqrstvwxyz'])

def count_unique_characters(word):
    return len(set(word))

def count_punctuation(sentence):
    return len([char for char in sentence if char in '.,!?;:'])

def count_stopwords(sentence):
    stopwords = spacy.lang.en.stop_words.STOP_WORDS
    return len([word for word in sentence.split() if word.lower() in stopwords])

def get_bert_embedding(word):
    inputs = tokenizer(word, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

def count_consonant_groups(word):
    consonants = "bcdfghjklmnpqrstvwxyz"
    groups = 0
    group_length = 0
    
    for char in word.lower():
        if char in consonants:
            group_length += 1
        else:
            if group_length > 1:
                groups += 1
            group_length = 0
    
    # Check if the last group at the end of the word was a valid consonant group
    if group_length > 1:
        groups += 1
    
    return groups

def extract_features(df):
    features = []
    
    for index, row in df.iterrows():
        word = row['word']
        sentence = row['sentence']
        
        #Word-level features
        word_freq = word_frequency(word, 'en')
        word_length = len(word)
        syllable_count = len([char for char in word if char.lower() in 'aeiou'])
        vowel_count = sum([1 for char in word if char.lower() in 'aeiou'])
        consonant_count = count_consonants(word)
        unique_char_count = count_unique_characters(word)
        consonant_groups_count = count_consonant_groups(word)  # Added feature
        
        
       # Sentence-level features
        doc = nlp(sentence)
        sentence_length = len(sentence.split())
        avg_word_length = np.mean([len(w) for w in sentence.split()])
        unique_word_count = len(set(sentence.split()))
        punctuation_count = count_punctuation(sentence)
        stopword_count = count_stopwords(sentence)
        sentence_embedding = nlp(sentence).vector
        
        is_title = int(word.istitle())
        is_entity = int(any(ent.text == word for ent in doc.ents))
        syntactic_relations = len([tok.dep_ for tok in doc if tok.text == word])
        
        word_embedding = nlp(word).vector
        word_freq_in_sentence = sentence.lower().split().count(word.lower())

       #Semantic features
        pos = nlp(word)[0].pos_
        synsets_count = len(wn.synsets(word))
        
        synset_depth = max([synset.min_depth() for synset in wn.synsets(word)], default=0)
        hypernyms_count = sum([len(synset.hypernyms()) for synset in wn.synsets(word)])
        hyponyms_count = sum([len(synset.hyponyms()) for synset in wn.synsets(word)])
        meronyms_count = sum([len(synset.part_meronyms()) for synset in wn.synsets(word)])
        
        bert_embedding = get_bert_embedding(word)
        
        # Additional semantic and psycholinguistic features
        synonyms_count = sum([len(synset.lemma_names()) for synset in wn.synsets(word)])
        antonyms_count = sum([len(lemma.antonyms()) for synset in wn.synsets(word) for lemma in synset.lemmas()])
        
        # Feature vector construction
        feature_vector = [
            word_freq, word_length,  vowel_count, consonant_count, unique_char_count,
            consonant_groups_count,  # Included the new feature
             is_entity, syntactic_relations, word_freq_in_sentence, synsets_count, 
            hypernyms_count, hyponyms_count, meronyms_count, synonyms_count, antonyms_count,
            syllable_count,synset_depth ,sentence_length, avg_word_length, unique_word_count, punctuation_count, stopword_count,is_title,
        ] + list(word_embedding) + list(sentence_embedding)   + list(bert_embedding)
        
        #POS features
        pos_features = [0] * 17
        pos_mapping = {
            'ADJ': 0, 'ADP': 1, 'ADV': 2, 'AUX': 3, 'CCONJ': 4,
            'DET': 5, 'INTJ': 6, 'NOUN': 7, 'NUM': 8, 'PART': 9,
            'PRON': 10, 'PROPN': 11, 'PUNCT': 12, 'SCONJ': 13,
            'SYM': 14, 'VERB': 15, 'X': 16
        }
        if pos in pos_mapping:
            pos_features[pos_mapping[pos]] = 1
        
        feature_vector += pos_features
        features.append(feature_vector)
    
    return np.array(features)
    
    
# Extracția caracteristicilor pentru datele de antrenament și test
X_train = extract_features(train_df)
y_train = train_df['complexity'].values
X_test = extract_features(test_df)


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Antrenarea modelului
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_split, y_train_split)

# Predicții pe setul de validare
y_pred = model.predict(X_valid_split)

# Evaluarea modelului
r2 = r2_score(y_valid_split, y_pred)
pearson_corr, _ = pearsonr(y_valid_split, y_pred)
metric = 0.5 * (abs(pearson_corr) + max(0, r2))

print(f'R^2: {r2}')
print(f'Pearson Correlation: {pearson_corr}')
print(f'Metric: {metric}')

test_pred = model.predict(X_test)

# Generarea fișierului de submitere
submission_df = pd.DataFrame({
    'cur_id': test_df['cur_id'],
    'complexity': test_pred
})

submission_df.to_csv('submission.csv', index=False)


