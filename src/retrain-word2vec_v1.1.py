# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 10:41:51 2018

@author: Krishna
"""

import sys
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phraser, Phrases
import os
import elasticsearch.helpers
from elasticsearch import Elasticsearch
from nltk import ngrams
import nltk
nltk.download('punkt')

def get_text(es_index_name, es_index_doctype, index_ports):
    
    es = Elasticsearch(port = index_ports)
    corpus = []
    
    res = elasticsearch.helpers.scan(es,
                                     query = {"query": {"match_all": {}}},
                                     index = es_index_name,
                                     doc_type = es_index_doctype)
    results_data = list(res)
    
    for i in range(len(results_data)):
        corpus.append(results_data[i]['_source']['body'])
        
    return corpus

def check_1(s):
    return any(i.isdigit() for i in s)

def check_2(s):
    return all(i in string.punctuation for i in s)

def generate_sent_tokens(corpus, n_ngrams):
    punctuations = set(string.punctuation).union(set(("``", "''")))
    tokenized_corpus = []
    
    for text in corpus:
        tok_text = word_tokenize(text.lower())
        clean_text = ' '
        for word in tok_text:
            if word not in punctuations and not check_2(word):
                clean_text += word + ' '
        tokenized_sentences = list(map(list, (ngrams(clean_text.split(), n_ngrams))))
        if len(tokenized_sentences) == 0:
            tokenized_sentences = [clean_text.split()]
        tokenized_corpus.extend(tokenized_sentences)
    
    # Phrase Detection
    # Give some common terms that can be ignored in phrase detection
    # For example, 'state_of_affairs' will be detected because 'of' is provided here: 
    common_terms = ["of", "with", "without", "and", "or", "the", "a"]
    # Create the relevant phrases from the list of sentences:
    phrases = Phrases(tokenized_corpus, common_terms=common_terms)
    # The Phraser object is used from now on to transform sentences
    bigram = Phraser(phrases)
    # Applying the Phraser to transform our sentences is simply
    tokenized_corpus = list(bigram[tokenized_corpus])
    
    return tokenized_corpus


def retrain_word2vec(pretrained_word2vec_file, tokenized_corpus, word2vec_retrain_epochs, start_alpha, end_alpha):
    
    model = Word2Vec(size = 300, min_count = 1)
    model.build_vocab(tokenized_corpus)
    model.intersect_word2vec_format(pretrained_word2vec_file, lockf = 1.0, binary = True)
    
    model.train(tokenized_corpus,
                total_examples = model.corpus_count,
                epochs = word2vec_retrain_epochs,
                start_alpha = start_alpha,
                end_alpha = end_alpha)
    
    return model


if __name__ == '__main__':
    
    inputdata = sys.argv
    
    es_index_name = inputdata[1]
    es_index_ngram_matchscoretype = inputdata[2]
    es_index_doctype = inputdata[3]    
    pretrained_word2vec_file = inputdata[4]
    word2vec_retrain_epochs = int(inputdata[5])
    start_alpha = float(inputdata[6])
    end_alpha = float(inputdata[7])
    output_folderpath = inputdata[8]
    n_ngrams = int(inputdata[9])
    index_ports = int(inputdata[10])
    
    corpus = get_text(es_index_name, es_index_doctype, index_ports)
    tokenized_corpus = generate_sent_tokens(corpus, n_ngrams)
    model = retrain_word2vec(pretrained_word2vec_file, tokenized_corpus, word2vec_retrain_epochs, start_alpha, end_alpha)
    
    output_foldername = output_folderpath + 'retrained-word2vec.'+es_index_name+'.'+es_index_ngram_matchscoretype+'.epochs='+str(word2vec_retrain_epochs)+ os.sep
    
    if not os.path.exists(output_foldername):
        os.makedirs(output_foldername)
        
    model.save(output_foldername + 'retrained-word2vec.model')
    print ("\nRetrained Word2vec model present at", output_foldername)