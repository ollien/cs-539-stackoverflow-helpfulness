import click
import pandas
import sklearn.feature_extraction.text
import sklearn.naive_bayes
import sklearn.metrics
import nltk
import numpy
import math
from sklearn.ensemble import AdaBoostClassifier


def get_file_data(filename: str):
    all_data = pandas.read_csv(filename)
    return all_data[["BodyCleaned", "Y"]]


def vectorize_data(data):
    stop_words = nltk.corpus.stopwords.words("english")
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words=stop_words)
    counts = vectorizer.fit_transform(data)
    transformer = sklearn.feature_extraction.text.TfidfTransformer().fit(counts)
    return transformer.transform(counts)

def nltk_download():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

@click.command()
@click.option("--train_file", required=True)
@click.option("--test_file", required=True)
def main(train_file: str, test_file: str):
    nltk_download()
    training_data = get_file_data(train_file)
    test_data = get_file_data(train_file)
    res = vectorize_data(training_data["BodyCleaned"])
    model = AdaBoostClassifier(n_estimators=5000, random_state=0)
    model.fit(res, training_data["Y"])
    vectorized_test_data = vectorize_data(training_data["BodyCleaned"])
    predicted = model.predict(vectorized_test_data)
    print(sklearn.metrics.accuracy_score(training_data["Y"], predicted))


if __name__ == "__main__":
    main()



