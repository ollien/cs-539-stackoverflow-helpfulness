import click
import pandas


@click.command()
@click.argument("in_file")
@click.argument("out_file")
def main(in_file: str, out_file: str):
    data = pandas.read_csv(in_file)
    to_write = data[["num", "BodyCleaned", "Y"]]
    to_write["Y"] = to_write["Y"].map({"HQ": 0, "LQ_CLOSE": 1, "LQ_EDIT": 2})

    to_write.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
