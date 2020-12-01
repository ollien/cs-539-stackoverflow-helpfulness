import pandas
import numpy
import sklearn, sklearn.model_selection, sklearn.linear_model, sklearn.metrics


def main(nn_file_path: str, other_classifier_file: str):
    nn_file = pandas.read_csv(nn_file_path)
    sklearn_file = pandas.read_csv(other_classifier_file)
    sklearn_file["nn"] = numpy.nan
    for i in range(max(nn_file["num"]) + 1):
        res = nn_file.loc[nn_file["num"] == i]
        if len(res) == 0:
            sklearn_file = sklearn_file.drop(sklearn_file[sklearn_file.num == i].index)
            print("Dropping", i)
            continue
        prediction = res.iloc[0]["predicted"]
        sklearn_file.loc[sklearn_file.num == i, "nn"] = prediction

    print("Training...")
    trainX, testX, trainY, testY = sklearn.model_selection.train_test_split(
        sklearn_file[["lr", "cart", "svc", "knn", "nn"]],
        sklearn_file["Y"],
        test_size=0.33,
    )

    logistic = sklearn.linear_model.LogisticRegression()
    logistic.fit(trainX, trainY)
    print("Testing...")
    predictions = logistic.predict(testX)

    return sklearn.metrics.accuracy_score(testY, predictions)


if __name__ == "__main__":
    main("nn_out.csv", "stack_out.csv")