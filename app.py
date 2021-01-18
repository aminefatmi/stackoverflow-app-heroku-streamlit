# import numpy as np
# from flask import Flask, request, jsonify, render_template
import pickle
import streamlit as st
# import pandas as pd
from bs4 import BeautifulSoup
import re
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import 	WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# app = Flask(__name__)
OVR_LinearSVC = pickle.load(open(r'ovr_linearsvc_op1.pkl', 'rb'))
mlb = pickle.load(open(r'mlb_svc.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open(r'tfidf_vectorizer_svc.pkl', 'rb'))


stop_minus_tags = pickle.load(open(r'stop_minus_tags.pkl', 'rb'))
top_400_tags = pickle.load(open(r'top_400_tags.pkl', 'rb'))
single_word_top_400_tags = pickle.load(open(r'single_word_top_400_tags.pkl', 'rb'))
# class definition to remove contractions

R_patterns = [
    (r'won\'t', 'will not'),
    (r'can\'t', 'cannot'),
    (r'[Ii]\'m', 'i am'),
    (r'(\w+)\'ll', '\g<1> will'),
    (r'(\w+)n\'t', '\g<1> not'),
    (r'(\w+)\'ve', '\g<1> have'),
    (r'(\w+)\'s', '\g<1> is'),
    (r'(\w+)\'re', '\g<1> are'),
]

class REReplacer(object):
    def __init__(self, pattern = R_patterns):
      self.pattern = [(re.compile(regex), repl) for (regex, repl) in pattern]
    def replace(self, text):
      s = text
      for (pattern, repl) in self.pattern:
          s = re.sub(pattern, repl, s)
      return s

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

mwtokenizer = nltk.MWETokenizer(separator='')
mwtokenizer.add_mwe(('c', '#'))

def clean_body_title(text):
    
  # parse html
  soup = BeautifulSoup(text, features="html.parser")

  # kill all script and style elements
  for script in soup(["script", "style"]):
      script.extract()    # rip it out

  # get text
  text = soup.get_text()

  # remove url
  text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

  # convert to lowercase
  text = text.lower()

  # remove contractions
  rep_word = REReplacer()
  text = rep_word.replace(text)
  
  # split into words
  mwtokenizer = nltk.MWETokenizer(separator='')
  mwtokenizer.add_mwe(('c', '#')) 
  tokens = mwtokenizer.tokenize(word_tokenize(text))

  # remove punctuation from each word
  table = str.maketrans('', '', string.punctuation)
  stripped=[]
  for w in tokens:
    if w in set(single_word_top_400_tags):
      stripped.append(w)
    else:
      stripped.append(w.translate(table))

  # remove remaining tokens that are not alphabetic
  words = [word for word in stripped if (any(chr.isdigit() for chr in word)==False)]

  # filter out stop words
  stop_words = set(stopwords.words('english'))
  words = [w for w in words if not w in stop_words]
  words = [w for w in words if w != '']

  # lemmatization except top 400 tags
  wordnet_lemmatizer = WordNetLemmatizer()
  words_lemma = []
  for w in words:
    if w in set(single_word_top_400_tags):
      words_lemma.append(w)
    else:
      words_lemma.append(wordnet_lemmatizer.lemmatize(w, get_wordnet_pos(w)))

  return words_lemma

# The result of this function is what we input in the different models

def pipeline_title_body(title, body):
  # clean body and title
  title = clean_body_title(title)
  body = clean_body_title(body)

  # join body and title lists
  body = ' '.join(body)
  title = ' '.join(title)

  # concatenate cleaned body and title
  body_title = body + ' ' + title

  # tokenize body, title and body_title
  body_title_token = mwtokenizer.tokenize(word_tokenize(body_title))
  body_token = mwtokenizer.tokenize(word_tokenize(body))
  title_token = mwtokenizer.tokenize(word_tokenize(title))

  # filter body_title by removing stop words defined manually in list_stop_words
  body_title_token_stops = [w for w in body_title_token if w not in stop_minus_tags]

  # filter body_title by keeping only words that are tags or both in title and body
  body_title_token_stops_filter = [word for word in body_title_token_stops if ((word in single_word_top_400_tags) \
                                                                or (word in body_token and word in title_token))]

  # join the list
  body_title_token_stops_filter = ' '.join(body_title_token_stops_filter)

  return body_title_token_stops_filter

def tag_supervised(body_title_cleaned):
    X_tfidf = tfidf_vectorizer.transform([body_title_cleaned])
    prediction = OVR_LinearSVC.predict(X_tfidf)
    tags = mlb.inverse_transform(prediction)
    
    return tags


st.title('''
Stackoverflow tag prediction       
''')

question = st.text_input("Enter your question title")
body = st.text_input("Enter your question body")

# input_text = [x for x in request.form.values()]
# question = input('Enter a question: ')
# body = input('Enter a body: ')
raw_cleaned = pipeline_title_body(question, body)
prediction = tag_supervised(raw_cleaned)
prediction = list(prediction[0])

st.markdown('<style>' + open('tag.css').read() + '</style>', unsafe_allow_html=True) 


if st.button('Predict'):
    if len(prediction)==0:
        st.text('No tag found')
    elif len(prediction)==1:
        st.markdown('''<p><span>&nbsp{}&nbsp</span></p>
            '''.format(prediction[0]), unsafe_allow_html=True)
    elif len(prediction)==2:
        st.markdown('''<p><span>&nbsp{}&nbsp</span> &nbsp 
                    <span>&nbsp{}&nbsp</span>
                    </p>
            '''.format(prediction[0],
            prediction[1]
            ), unsafe_allow_html=True)
    elif len(prediction)==3:
        st.markdown('''<p><span>&nbsp{}&nbsp</span> &nbsp 
                    <span>&nbsp{}&nbsp</span> &nbsp 
                    <span>&nbsp{}&nbsp</span>
                    </p>
            '''.format(prediction[0],
            prediction[1],
            prediction[2]
            ), unsafe_allow_html=True)
    elif len(prediction)==4:
        st.markdown('''<p><span>&nbsp{}&nbsp</span> &nbsp 
                    <span>&nbsp{}&nbsp</span> &nbsp 
                    <span>&nbsp{}&nbsp</span> &nbsp 
        <span>&nbsp{}&nbsp</span></p>
            '''.format(prediction[0],
            prediction[1],
            prediction[2],
            prediction[3]
            ), unsafe_allow_html=True)
    elif len(prediction)==5:
        st.markdown('''<p><span>&nbsp{}&nbsp</span> &nbsp 
                 <span>&nbsp{}&nbsp</span> &nbsp 
                    <span>&nbsp{}&nbsp</span> &nbsp 
                    <span>&nbsp{}&nbsp</span> &nbsp    
        <span>&nbsp{}&nbsp</span></p>
            '''.format(prediction[0],
            prediction[1],
            prediction[2],
            prediction[3],
            prediction[4]
            ), unsafe_allow_html=True)
