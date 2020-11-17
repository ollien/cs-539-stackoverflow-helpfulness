# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:42:07 2020

@author: Veronica
"""

import pandas as pd
import numpy as np
import re
import string

def remove_code(text):
    #print(text)
    split = text.split('<code>', 1)
    if len(split) == 1: return text
    split2 = split[1].split('</code>')
    if len(split2) == 1: return split[0]
    return split[0] + remove_code(split[1].split('</code>', 1)[1])

def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)  

#read csv files
df = pd.read_csv('train.csv')
df['Body'] = df['Body'].astype(str)
  
#get instances where number of <code> and </code> tags are the same
df['<code>'] = df['Body'].apply(lambda x: x.count('<code>'))
df['</code>'] = df['Body'].apply(lambda x: x.count('</code>'))
df['equal'] = df.apply(lambda x: x['<code>'] == x['</code>'], axis=1)
df2 = df[df['equal'] == True]
df2.drop(['<code>', '</code>', 'equal'], axis=1, inplace=True)

#drop code
df2['BodyCleaned'] = df2['Body'].apply(lambda x: remove_code(x))
    
#drop tags
df2['BodyCleaned'] = df2['BodyCleaned'].apply(lambda x: remove_tags(x))
 
#remove line breaks
df2['BodyCleaned'] = df2['BodyCleaned'].apply(lambda x: x.replace('\r', ' ').replace('\n', ' '))

#remove punctuation
punc = string.punctuation.replace("'", "")
punc = string.punctuation.replace(".", "")
df2['BodyCleaned'] = df2['BodyCleaned'].apply(lambda x: x.translate(str.maketrans('', '', punc)))

df2.to_csv('train-cleaned.csv') 