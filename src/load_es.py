# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:05:08 2018

@author: Krishna
"""

import string
from nltk.tokenize import sent_tokenize, word_tokenize
from elasticsearch import Elasticsearch 
from sklearn.datasets import fetch_20newsgroups
import random

def check_1(s):
    return any(i.isdigit() for i in s)

def check_2(s):
    return all(i in string.punctuation for i in s)


def generate_sent_tokens(corpus):
    punctuations = set(string.punctuation).union(set(("``", "''")))
    tokenized_corpus = []
    sentence = ''
    
    for text in corpus:
        sentences = sent_tokenize(text.lower())
        for sent in sentences:
            temp = []
            for s in word_tokenize(sent):
                if s not in punctuations and not check_1(s) and not check_2(s):
                   temp.append(s) 
            tokenized_corpus.append(temp)
    
    for i in range(len(tokenized_corpus)):
        sentence += ' '.join(tokenized_corpus[i])
        sentence += '. '
    
    return sentence


NG_all = fetch_20newsgroups(subset = 'all',
                              remove = ('headers', 'footers', 'quotes'))

data = NG_all.data
labels = NG_all.target


es = Elasticsearch()
doc_id = 0
doc_ids = []
for i in range(len(data)):
    body = generate_sent_tokens([data[i]])
    #body = data[i]
    body_field_length = len(body.split())
    es_doc = {
              'doc_id': doc_id,
              'body': body,
              'body_field_length': body_field_length,
              'random': random.uniform(0,1),
              #'label_array': [NG_all.target_names[labels[i]]]
              'label_array': [int(labels[i])]
              }
    res = es.index(index = "20ng_all",
                   doc_type = "document",
                   id = doc_id,
                   body = es_doc)
    if res['created'] == True:
        doc_ids.append(doc_id)
    doc_id += 1
    print(res['created'])