---
title: "Low preforming models"
---

***

##### Concatenating meta data values-mid NN architecture #####

One of the models that we used that did not turn out to have good preformance was one where after the post was sent through the GRU layer, 
we concatenated on the result from the GRU with the meta data values such as the text to code ratio or the amount of text written. 
From here, we passed this new concatonated tensor into the multiple fully connected layers.  

##### Preformance #####

The preformance for this model was not very good, with it being around 45% accuracy even after hyperparameter tuning and changing layer count.  
One way that we could improve this is by finding more or different metadata values that we could concatonate onto the end of the result from the GRU layer, 
in order to see which metadata values have a greater effect on the training of the model.

***	

##### Custom Word2Vec architecture #####

Another model we used was that we replaced the pre-trained fasttext embedding with a word2vec embedding that was built from the stackoverflow posts, 
and then used these word embeddings to train a neural net model with a RNN.


##### Preformance #####
The preformance for this model was not very good, with having being around 35% accuracy.  
We suspect that this model has a very low accuraccy due to the small set of data that the word2vec algorithim had to be able to build the word embeddings.  
If we had extended time, we may have been able to scrape even more stackoverflow posts in order to get a larger set of data for the word2vec embeddings to be made from.