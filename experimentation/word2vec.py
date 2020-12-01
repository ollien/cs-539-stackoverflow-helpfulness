from gensim.models import Word2Vec
import pandas as pd

def remove_words(line, words):
    return list(filter(lambda x: x in words, line))

df = pd.read_csv('train-cleaned.csv')
df['BodyCleaned'] = df['BodyCleaned'].astype(str)
df['BodyCleaned'] = df['BodyCleaned'].apply(lambda x: x.split())

sentences = df['BodyCleaned'].apply(list)

model = Word2Vec(sentences, 
                 min_count=3,   
                 size=10,     
                 workers=2,
                 window=5,
                 iter=100)

model.wv.save_word2vec_format('./model.bin', binary=True)

model.wv.save_word2vec_format('./model.txt', binary=False)

#drop words not in model
words = list(model.wv.vocab)
df['BodyCleaned'] = df['BodyCleaned'].apply(lambda x: remove_words(x, words))

#drop rows with zero words
df['len'] = df['BodyCleaned'].apply(lambda x: len(x))
df = df[df['len'] != 0]
df.drop('len', axis=1, inplace=True)

#vectorize
df['BodyVectorized'] = df['BodyCleaned'].apply(lambda x: model[x])

#save new csv
df.to_csv('train-vectorized.csv')