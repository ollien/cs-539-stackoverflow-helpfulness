# -*- coding: utf-8 -*-
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import collections
import csv

# from google.colab import files
# import io
import pandas as pd

DEFAULT_TRAIN_FILE = "3_fold/train_0.csv"
DEFAULT_TEST_FILE = "3_fold/test_0.csv"
DEFAULT_OUT_FILE = "stack_out.csv"
CLASS_MAP = {"HQ": 0, "LQ_CLOSE": 1, "LQ_EDIT": 2}


# Upload the dataset
# Clean columns for null values
# Select training variables
# Select labels


def get_data(filename):

    """Colab Version
    uploaded = files.upload()
    train_file = pd.read_csv(io.BytesIO(uploaded['train.csv']))"""

    data = pd.read_csv(filename)
    data = data[data.asker_reputation != 0]
    data = data[data.views != 0]

    data["asker_creation_date"].fillna(0, inplace=True)

    data["asker_reputation"].fillna(0, inplace=True)

    data["views"].fillna(0, inplace=True)

    data["Text-Code Ratio"].fillna(0, inplace=True)

    data["Text"].fillna(0, inplace=True)

    data["Code"].fillna(0, inplace=True)
    data["Asker_Question_Year"].fillna(0, inplace=True)

    X = data.loc[
        :,
        [
            "asker_reputation",
            "views",
            "Text-Code Ratio",
            "Text",
            "Code",
            "Asker_Question_Year",
        ],
    ].values

    y = data.loc[:, ["Y"]].values

    return X, y


# Define machine learning models


# get a list of models to evaluate
def get_models():
    models = dict()
    models["lr"] = LogisticRegression()
    models["knn"] = KNeighborsClassifier()
    models["cart"] = DecisionTreeClassifier()
    models["svc"] = SVC()
    return models


def main(train_file: str, test_file: str, out_file: str):
    # define dataset
    X, y = get_data(train_file)
    testX, testY = get_data(test_file)
    # get the models to evaluate
    models = get_models()
    # evaluate the models and store results
    results, model_names = list(), list()
    for name, model in models.items():
        print(f"Training {name}")
        model.fit(X, y)

    predictions = collections.defaultdict(list)
    predictions["Y"] = list(testY.squeeze())
    for name, model in models.items():
        predictions[name] = model.predict(testX)

    with open(out_file, "w") as out:
        keys = list(predictions.keys())
        writer = csv.writer(out)
        writer.writerow(["num"] + keys)
        for i in range(len(testY)):
            # I am so sorry to whoever reads this
            to_write = [CLASS_MAP[predictions[key][i]] for key in keys]
            to_write = [i] + to_write
            writer.writerow(to_write)


if __name__ == "__main__":
    main(DEFAULT_TRAIN_FILE, DEFAULT_TEST_FILE, DEFAULT_OUT_FILE)
