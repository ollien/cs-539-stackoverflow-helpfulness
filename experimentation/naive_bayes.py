import click
import pandas
import sklearn.feature_extraction.text
import sklearn.naive_bayes
import sklearn.metrics
import nltk
import numpy
import math


def get_file_data(filename: str):
    all_data = pandas.read_csv(filename)
    return all_data[["BodyCleaned", "Y"]]


def vectorize_data(data):
    stop_words = nltk.corpus.stopwords.words("english")
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        # If the item appears in at least five documents, use it
        stop_words=stop_words, min_df=5 / 45000
    )
    counts = vectorizer.fit_transform(data)
    transformer = sklearn.feature_extraction.text.TfidfTransformer().fit(counts)
    return vectorizer, transformer, transformer.transform(counts)


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
    test_data = get_file_data(test_file)
    vectorizer, transformer, res = vectorize_data(training_data["BodyCleaned"])
    model = sklearn.naive_bayes.MultinomialNB()
    model.fit(res, training_data["Y"])
    vectorized_test_data = transformer.transform(
        vectorizer.transform(test_data["BodyCleaned"])
    )
    predicted = model.predict(vectorized_test_data)
    print(sklearn.metrics.accuracy_score(test_data["Y"], predicted))


if __name__ == "__main__":
    main()
