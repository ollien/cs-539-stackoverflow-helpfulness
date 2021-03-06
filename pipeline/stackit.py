import pandas
import numpy
import sklearn, sklearn.model_selection, sklearn.linear_model, sklearn.metrics, sklearn.svm


def main(nn_file_path: str, other_classifier_file: str):
    nn_file = pandas.read_csv(nn_file_path)
    sklearn_file = pandas.read_csv(other_classifier_file)
    sklearn_file["nn"] = numpy.nan
    for i in range(max(nn_file["num"]) + 1):
        res = nn_file.loc[nn_file["num"] == i]
        if len(res) == 0:
            # If something wasn't trained by the neural net due to batch size constraints, it may have to be dropped as data in the ensemble
            sklearn_file = sklearn_file.drop(sklearn_file[sklearn_file.num == i].index)
            print("Dropping", i)
            continue
        prediction = res.iloc[0]["predicted"]
        sklearn_file.loc[sklearn_file.num == i, "nn"] = prediction

    print("Training...")
    logistic = sklearn.svm.SVC(C=4, kernel="rbf")
    crossval_scores = sklearn.model_selection.cross_validate(
        logistic,
        sklearn_file[["lr", "cart", "svc", "knn", "nn"]],
        sklearn_file["Y"],
        cv=5,
        scoring="accuracy",
    )

    return max(crossval_scores["test_score"])


if __name__ == "__main__":
    main("nn_out.csv", "stack_out.csv")
