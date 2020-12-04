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

DEFAULT_TRAIN_FILE = "all_data.csv"
DEFAULT_TEST_FILE = "all_data.csv"
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
    data["Y"].fillna(0, inplace = True)

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

# get a list of models to evaluate
def get_models():
    models = dict()
    models["lr"] = LogisticRegression(max_iter = 900)
    models["knn"] = KNeighborsClassifier(n_neighbors = 1)
    models["cart"] = DecisionTreeClassifier(random_state=0, max_depth = 600)
    models["svc"] = SVC(C=4)
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

    index = 0
    true = 0
    false = 0


    for name, model in models.items():
        predictions[name] = model.predict(testX)

    Lr_Pos = 0
    Knn_Pos = 0
    cart_Pos = 0
    svc_Pos = 0
    with open(out_file, "w") as out:
        keys = list(predictions.keys())
        writer = csv.writer(out)
        writer.writerow(["num"] + keys)
        for i in range(len(testY)):
            # I am so sorry to whoever reads this
            to_write = [CLASS_MAP[predictions[key][i]] for key in keys]
            to_write = [i] + to_write
            if to_write[1] == to_write[2]:
                Lr_Pos +=1
            else:
               continue
            if to_write[1] == to_write[3]:
                Knn_Pos+=1
            else:
                continue
            if to_write[1] == to_write[4]:
                cart_Pos+=1
            else:
                continue
            if to_write[1] == to_write[5]:
                svc_Pos+=1
            else:
                continue
            writer.writerow(to_write)

    print("Lr: ", Lr_Pos/len(testY))
    print("Knn: ", Knn_Pos/len(testY))
    print("cart: ", cart_Pos/len(testY))
    print("svc: ", svc_Pos/len(testY))
if __name__ == "__main__":
    main(DEFAULT_TRAIN_FILE, DEFAULT_TEST_FILE, DEFAULT_OUT_FILE)