---
title: "Approaches Used"
---

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

***

##### RNN - Recurrent Neural Network #####

This was our originally proposed approach, since recurrent neural networks are generally good at text classification since they can handle context for each post. Word vectorization on each post's body was performed using a premade [FastText](https://fasttext.cc/) model.
The vectorized posts were fed into a GRU, (gated recurrent unit), followed by a fully connected layer, for classification.
The neural network itself was implemented using torch.

Accuracy: ~76%

***

##### Stacking Ensemble #####

For this approach, we trained all of the above models for Naive Bayes, RNN, and the metadata classifiers in same manner as before.
The results of all of these classifiers were combined as input to a stacking layer to use for its own classification, (treating the results of each of the previous models as a feature for each post).
The stacking layer originally used logistic regression, but using an SVC, (Support Vector Classifier) instead was found to give higher accuracy.

Accuracy: ~89% with a SVC stacking layer, ~86% with a Logistic Regression stacking layer.
