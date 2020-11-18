from gensim.models import Word2Vec
import pandas as pd

df = pd.read_csv('train-cleaned.csv')
df['BodyCleaned'] = df['BodyCleaned'].astype(str)
df['BodyCleaned'] = df['BodyCleaned'].apply(lambda x: x.split())

sentences = df['BodyCleaned'].apply(list)

model = Word2Vec(sentences, 
                 min_count=3,   
                 size=200,     
                 workers=2,
                 window=5,
                 iter=100)

model.wv.save_word2vec_format('./model.bin', binary=True)

model.wv.save_word2vec_format('./model.txt', binary=False)