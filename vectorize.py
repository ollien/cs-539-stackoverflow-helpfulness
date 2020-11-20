# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:10:34 2020

@author: Veronica
"""

import pandas as pd
import pickle
from gensim.models import Word2Vec
from gensim.models import keyedvectors

def remove_words(line, words):
    return list(filter(lambda x: x in words, line))

#load csv
df = pd.read_csv('train-cleaned.csv')

#load trained word2vec model
model = keyedvectors.KeyedVectors.load_word2vec_format('model.bin', binary=True)

#drop words not in model
df['BodyCleaned'] = df['BodyCleaned'].astype(str)
df['BodyCleaned'] = df['BodyCleaned'].apply(lambda x: x.split())
words = list(model.wv.vocab)
df['BodyCleaned'] = df['BodyCleaned'].apply(lambda x: remove_words(x, words))

#drop rows with zero words
df['len'] = df['BodyCleaned'].apply(lambda x: len(x))
df = df[df['len'] != 0]
df.drop('len', axis=1, inplace=True)

#vectorize
df['BodyVectorized'] = df['BodyCleaned'].apply(lambda x: model[x])
pickle.dump(df['BodyVectorized'].tolist(), 'bodyVectorized.pkl')
#save new csv
df.to_csv('train-vectorized.csv')