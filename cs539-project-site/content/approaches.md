---
title: "Methods and Results"
---

***

## Data Preparation ##

***

##### Data Scraping #####

The initial dataset from Kaggle had the question Id, title of the question, body of the question, tags, date the question was created, and the quality 
classification of the question (label). After formatting the original to feed question Id’s, we used the Stack Exchange API to scrape other metadata, 
such as asker reputation, number of views and the date the asker made their account on Stack Overflow. After cleaning the question body, we created some
other metadata, such as Text-Code Ratio, total characters of text and total characters of code in each post and the number of years the asker has had their account 
on Stack Overflow when they asked the question. 

***

##### Post Body Cleaning #####

Prior to vectorizing the bodies of the posts, the text was cleaned. HTML tags and code were removed. In addition, line breaks and punctuation were removed, 
and all the text was converted to lowercase so that the same word with different cases would not be interpreted to be a different word 
(for example, “Python” versus “python”).

****

## Main Approaches ##

***

##### Naive Bayes #####

We used TF-IDF for word embedding on each post's body, which was used as input for a scikit Naive Bayes model.
This model was used as a baseline for comparison to the more advanced approaches implemented later, and was not as accurate as the other approaches.

Accuracy: ~72.2%

***

##### Metadata Classifiers #####

While the original dataset had some additional information beyond the post body, in order to obtain more
useful features, the team performed some web scraping to get additional data for each of the posts.
These metadata classifiers each used the scraped features for the dataset, (asker reputation, questions views, etc.), and were trained independently.

The following scikit models were tested:
- LogisticRegression, Accuracy: ~80.5%
- KNeighborsClassifier, Accuracy: ~80.5%
- DecisionTreeClassifier, Accuracy: ~80.4%
- SVC, Accuracy: ~76.7%

With that said, it's worth asking: what makes a good question? 
	
With the models we ran, the most useful metadata was numerical values we collected or calculated – particularly asker reputation, number of views, 
text-code ratio, number of text characters, number of code characters, and the number of years the asker had their account on Stack Overflow. While each of 
these pieces of metadata contributed to improving the accuracy of the models, the two that showed the most significant trend are number of views and number of 
years the asker had their account, followed by a less significant but still relevant asker reputation. The data below shows the averages of the metadata for each
quality classification. The number of views shows a significant difference between a high quality question and either bad quality classification. In addition, 
even between edit and close, close has a slightly lower average number of views. Asker reputation and "asker question year" show some trends, but with less 
linearity. What we can see is that the high quality questions were made by askers that had very high reputation on Stack Overflow and had their account for 
longer. We can see from the data below that the calculations of Text-Code Ratio has less obvious trends and relationships to quality, though they did help 
improve the accuracy of our models by small percentages.

|                     | High Quality | Bad Quality - Edit | Bad Quality - Close |
|---------------------|--------------|--------------------|---------------------|
| Asker Reputation    | 7861.96  | 467.61         | 968.18          |
| Views               | 19934.39  | 430.84         | 428.02           |
| Text-Code Ratio     | 0.70     | 1.13           | 0.70            |
| Text                | 307.52     | 1033.52        | 247.75            |
| Code                | 755.34    | 5.45           | 504.61            |
| Asker Question Year | 3.51      | 1.04           | 1.50              |

***

##### Recurrent Neural Network (RNN) #####

This was our originally proposed approach, since recurrent neural networks are generally good at text classification since they can handle context for each post. Word vectorization on each post's body was performed using a premade [FastText](https://fasttext.cc/) model.
The vectorized posts were fed into a GRU, (gated recurrent unit), followed by a fully connected layer, for classification.
The neural network itself was implemented using torch.

Accuracy: ~76%

***

##### Stacking Ensemble #####

For this approach, we trained both the RNN and the metadata classifiers with a 5-fold cross validation.
The results of all of these classifiers were combined as input to a stacking layer to use for its own classification, (treating the results of each of the previous models as a feature for each post).
The stacking layer originally used logistic regression, but using an SVC, (Support Vector Classifier) instead was found to give higher accuracy.

Accuracy: ~89% with a SVC stacking layer, ~86% with a Logistic Regression stacking layer.

***

## Low-Performing Approaches ##

***

##### Concatenating metadata features with GRU results #####

One of the attempted approaches that had low performance was one where the result from the GRU layer were concatenated with the metadata features and
sent through multiple fully connected layers. We tried tuning the hyperparameters and changing the number of layers in the model but were not able
to improve the accuracy significantly. If we had more time, we could have created more features from metadata to try to improve the performance of this approach.

Accuracy: ~45%

***	

##### Custom Word2Vec architecture #####

A Word2vec model was trained using the text from the bodies of the posts. The word embeddings from this model were used to train a recurrent neural network. We
suspect that this model had low performance because there was a relatively small set of data used to build the Word2vec dictionary. We could improve this by either 
training the model with more posts or using a pre-trained dictionary that contains more words.

Accuracy: ~35%
