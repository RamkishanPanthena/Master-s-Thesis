# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 16:26:50 2018

@author: Krishna
"""

import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phraser, Phrases
from gensim.models import KeyedVectors

filepath = "C:\\Users\\Krishna\\Desktop\\Data-Science\\Northeastern-University\\NEU\\Study-Material\\MS-Thesis\\"
folder = "results\\run_1543005042.slashdot.lr=0.001.epochs=250.regularization=True.reg.parameter=0.01.multi-label.pred_threshold=0.5\\"

input_folderpath = filepath + folder
wordvec_filename = filepath + "data\\word2vec\\GoogleNews-vectors-negative300.bin"

vocab_dict = np.load(input_folderpath + 'vocab_dict.npy').item()
features_list = np.load(input_folderpath + 'features_list.npy')
#word_list = np.load(input_folderpath + 'word_list.npy')
#words_removed_list = np.load(input_folderpath + 'words_removed_list.npy')
theta = np.load(input_folderpath + 'theta.npy')
weights = np.load(input_folderpath + 'weights.npy')
#theta2 = np.load(input_folderpath + 'theta2.npy')
#weights2 = np.load(input_folderpath + 'weights2.npy')
#all_pairs = np.load(input_folderpath + 'all_pairs.npy')
#word_pairs = np.load(input_folderpath + 'word_pairs.npy')

#words_removed_list[:30]



#model = Word2Vec.load(input_folderpath + "retrained-word2vec.model")
model = Word2Vec(size = 300, min_count = 1)
model.build_vocab([vocab_dict.keys()])
model.intersect_word2vec_format(wordvec_filename, lockf = 1.0, binary = True)

model.most_similar('citizen')
model.similar_by_vector('male')

theta[vocab_dict['femal']]
theta[vocab_dict['male']]

ind = np.argsort(theta[:,2])[::-1]
theta[:,2][ind]
features_list[ind]