import datetime
import pandas as pd
import numpy as np
import click
from pandas.core.reshape.reshape import stack
import stackscrape.stackscrape
from typing import Sequence

# Return lists of 100 Train Ids
def fetch_post_data(
    client: stackscrape.stackscrape.Client, train_list: Sequence[int], dataframe
):
    chunks = [train_list[x : x + 100] for x in range(0, len(train_list), 100)]
    i = 0
    tags = []
    asker_reputation = []
    views = []
    creation_dates = []
    for chunk in chunks:
        i += 1
        if i % 100 == 0:
            print(f"Processed id #{i}")

        info = client.get_question_info(list(chunk))
        consolidated_info = consolidate_list_of_dicts(info, "id")
        sorted_questions = []
        for question_id in chunk:
            question_info = consolidated_info.get(question_id)
            sorted_questions.append(question_info)

        tags.extend(
            [
                ",".join(question["tags"]) if question else None
                for question in sorted_questions
            ]
        )
        asker_reputation.extend(
            [
                question["reputation"] if question else None
                for question in sorted_questions
            ]
        )
        views.extend(
            [question["views"] if question else None for question in sorted_questions]
        )

        user_info = client.get_user_info(
            [
                question["asker_id"]
                for question in sorted_questions
                if question and question["asker_id"]
            ]
        )
        consolidated_user_info = consolidate_list_of_dicts(user_info, "id")
        creation_dates.extend(
            [
                format_unix_timestamp(
                    consolidated_user_info[question["asker_id"]]["creation_date"]
                )
                if question and question["asker_id"]
                else None
                for question in sorted_questions
            ]
        )

    dataframe["asker_reputation"] = asker_reputation
    dataframe["views"] = views
    dataframe["asker_creation_date"] = creation_dates

    return dataframe


def consolidate_list_of_dicts(dicts, id_key):
    res = {}
    for d in dicts:
        key = d[id_key]
        res[key] = d

    return res


def format_unix_timestamp(timestamp):
    timestamp_datetime = datetime.datetime.fromtimestamp(timestamp)
    return timestamp_datetime.strftime("%Y-%m-%d %H:%M:%S")


@click.command()
@click.argument("csv_file")
@click.argument("out_file")
def main(csv_file, out_file):
    client = stackscrape.stackscrape.Client()
    client.authenticate()
    dataframe = pd.read_csv(csv_file)
    ids_to_process = dataframe["Id"]
    dataframe = fetch_post_data(client, ids_to_process, dataframe)
    dataframe.to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
