---
title: "Project Overview"
---

***

##### Motivation #####

StackOverflow is a website where users can post computer programming related questions. It is important that questions have high quality
so that they will be helpful to both the user who posted the question and users with similar problems. While StackOverflow does check questions
for similarity to already existing questions, it does not have a tool to automatically assess question quality. There is a Triage system in place,
in which users can mark questions as low quality for further moderator review. However, this system relies on manual intervention and is prone to
bias, as different moderators may have different criteria regarding what makes a question of high quality.

Thus, the goal of this project was to develop a tool that will automatically assess the quality of StackOverflow questions. Such a tool would be more
efficient than the existing Triage system because it would not require manual intervention. In addition, it would be less biased because there would
be only one metric for evaluating question quality rather than several moderators. This tool would give users feedback on their question before it is
posted so they can make sure that it is of high enough quality to be useful to others.

***

##### Data Set #####

This project used a [data set](https://www.kaggle.com/imoore/60k-stack-overflow-questions-with-quality-rate) from Kaggle that consists of 60,000
StackOverflow questions from 2016-2020. The data set contains the question ID, title, body, related tags, and post date. Each question is labelled
with a quality, which can be one of the following:

1. HQ: High-quality posts with 30+ score and without a single edit.
2. LQ_EDIT: Low-quality posts with a negative score and with multiple community edits. However, they still remain open after the edits.
3. LQ_CLOSE: Low-quality posts that were closed by the community without a single edit.

The labels in this data set are balanced. In addition, since the data set contains the ID of the question, it is possible to go back and scrape
more data from the question if necessary.

If you would like to explore the data (with our augmentations mentioned in the approaches section), you can view it on [DoltHub](https://www.dolthub.com/repositories/ollien/cs-539-stackoverflow-data)

***

##### Proposed Approach ###

The team planned to use some form of a neural network for the core predictive algorithm. It was necessary to vectorize the body of the post prior
to feeding it into the neural network. Existing frameworks for word vectorization include [Word2vec](https://www.tensorflow.org/tutorials/text/word2vec)
and [GloVe](https://nlp.stanford.edu/projects/glove/). Another option was using additional features extracted from the questions to train other types
of classifiers, such as random forests, and combining these classifiers using an ensemble method.

#### Final Approach

After much experimentation, our final approach used a stacking ensemble of an RNN to read post bodies, and a combination of KNN, logistic regression, decision trees, and SVMs to read metadata. With this, we were able to produce a model that has an accuracy of about 89%.
