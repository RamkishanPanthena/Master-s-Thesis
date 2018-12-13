# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 21:17:27 2018

@author: Krishna
"""

import sys
from gensim.models.word2vec import Word2Vec

if __name__ == '__main__':
    
    inputdata = sys.argv
    
    pretrained_word2vec_folderpath = inputdata[1]
    test_word = inputdata[2]
    topn = int(inputdata[3])
    
    model = Word2Vec.load(pretrained_word2vec_folderpath + "retrained-word2vec.model")
    
    topwords = model.wv.most_similar(test_word, topn = topn)
    
    for i in range(len(topwords)):
        print(topwords[i])