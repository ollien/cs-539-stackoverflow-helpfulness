import os
import glob
import re
import click
from numpy.core.numeric import cross
import split_cv
import nn_preprocess
import torch_embedding
import other_classifiers
import stackit

CROSSVAL_DIRNAME = "crossval"
NEURALNET_PREFIX = "nn"
OUTPUT_DIR = "output"


def get_crossval_files(crossval_dir: str):
    files = glob.glob(os.path.join(crossval_dir, "*.csv"))
    train_files = [
        basename
        for file in files
        if (basename := os.path.basename(file)).startswith("train_")
    ]
    for train_file in train_files:
        test_file = re.sub(r"^train_", "test_", train_file)
        yield (train_file, test_file)


@click.command()
@click.option("--data_file", required=True)
@click.option("--n-splits", required=True, type=click.INT)
@click.option("--storage_dir", default="./processed_data")
def main(data_file: str, n_splits: int, storage_dir: str):
    print("Producing cross validation splits...")
    crossval_dir = os.path.join(storage_dir, "crossval")
    split_cv.run(n_splits, data_file, crossval_dir)

    print("Preprocessing files to pass into the neural network...")
    nn_dir = os.path.join(crossval_dir, "nn")
    os.makedirs(nn_dir)
    files_to_process = glob.glob(os.path.join(crossval_dir, "*.csv"))
    for file_to_process in files_to_process:
        filename = os.path.basename(file_to_process)
        out_filename = os.path.join(nn_dir, filename)
        nn_preprocess.run(file_to_process, out_filename)

    out_dir = os.path.join(storage_dir, OUTPUT_DIR)
    os.makedirs(out_dir)
    accuracies = []
    for i, (train_file, test_file) in enumerate(get_crossval_files(crossval_dir)):
        print(f"Running Crossval Iteration #{i}")
        print("Training neural network...")
        torch_embedding.main(
            os.path.join(nn_dir, train_file),
            os.path.join(nn_dir, test_file),
            os.path.join(out_dir, "nn_out.csv"),
        )

        print("Training other classifiers...")
        other_classifiers.main(
            os.path.join(crossval_dir, train_file),
            os.path.join(crossval_dir, test_file),
            os.path.join(out_dir, "other_out.csv"),
        )

        print("Training ensemble...")
        accuracy = stackit.main(
            os.path.join(out_dir, "nn_out.csv"),
            os.path.join(out_dir, "other_out.csv"),
        )

        print(accuracy)
        accuracies.append(accuracy)

    print("Best accuracy: ", max(accuracies))


if __name__ == "__main__":
    main()
