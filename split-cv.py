import click
import sklearn.model_selection
import pandas
import os
import errno
import csv


def make_dir(dir: str):
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


@click.command()
@click.argument("in_file")
@click.argument("out_dir")
@click.option("--n-splits", type=click.INT, required=True)
def main(n_splits: int, in_file: str, out_dir: str):
    """
    Convert a single CSV file into several k-fold split files in out_dir
    """
    make_dir(out_dir)
    data = pandas.read_csv(in_file)
    kfold = sklearn.model_selection.KFold(n_splits)
    indices = kfold.split(data)
    # I'm sorry to whoever has to read this code
    for split_no, (train, test) in enumerate(indices):
        print(f"Processing split #{split_no}")
        subsets = {"test": test, "train": train}
        for subset_name, subset in subsets.items():
            print(f"Starting {subset_name}, len: {len(subset)}")
            out_path = os.path.join(out_dir, f"{subset_name}_{split_no}.csv")
            with open(out_path, "w") as out_file:
                writer = csv.writer(out_file)
                writer.writerow(["num"] + list(data.columns))
                for i, item in enumerate(subset):
                    writer.writerow([i] + list(data.iloc[item]))


if __name__ == "__main__":
    main()
