import requests
import requests_oauthlib
import json
import ratelimit
from typing import Optional
import time
import datetime
import os

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "config.json")
# limit is actually 30/s, but SO seems to lock me out with anything over two
MAX_REQUESTS_PER_SECOND = 2


class Client:
    def __init__(self, config_path=CONFIG_FILE_PATH):
        self.client: Optional[requests_oauthlib.OAuth2Session] = None
        self.backoff_time: Optional[datetime.datetime] = None
        with open(config_path) as config_file:
            self.config = json.load(config_file)

    def authenticate(self):
        """
        Authenticate to stack exchange - must be done before making any requests
        """
        oauth = requests_oauthlib.OAuth2Session(
            self.config["client_id"], redirect_uri=self.config["redirect_uri"]
        )

        auth_url, state = oauth.authorization_url("https://stackoverflow.com/oauth")
        print(
            "Go to the following URL and log in. You will be redirected to a url starting with `localhost`. Copy and paste that url here"
        )
        print(auth_url)
        response_url = input("url: ")
        token = oauth.fetch_token(
            "https://stackoverflow.com/oauth/access_token",
            authorization_response=response_url,
            client_secret=self.config["client_secret"],
            include_client_id=True,
        )

        self.client = oauth

    def get_question_info(self, question_ids):
        if len(question_ids) > 100:
            raise ValueError(
                "StackExchange API only supports 100 question IDs at a time"
            )

        id_param = self._build_param_list(question_ids)
        # lol this is super gross - I could probably do some url building but this is a one-off scraping script
        url = f"https://api.stackexchange.com/2.2/questions/{id_param}?pagesize=100&site=stackoverflow&key={self.config['key']}"
        res = self._make_request("get", url)
        info = res.json()

        return [
            {
                "id": item["question_id"],
                "tags": item["tags"],
                "reputation": (
                    item["owner"]["reputation"]
                    if item["owner"]["user_type"] != "does_not_exist"
                    else None
                ),
                "asker_id": (
                    item["owner"]["user_id"]
                    if item["owner"]["user_type"] != "does_not_exist"
                    else None
                ),
                "views": item["view_count"],
            }
            for item in info["items"]
        ]

    def get_tag_info(self, tag_names):
        if len(tag_names) > 100:
            raise ValueError("StackExchange API only supports 100 tag names at a time")

        tag_param = self._build_param_list(tag_names)
        # lol this is super gross - I could probably do some url building but this is a one-off scraping script
        url = f"https://api.stackexchange.com/2.2/tags/{tag_param}/info?pagesize=100&site=stackoverflow&key={self.config['key']}"
        res = self._make_request("get", url)
        info = res.json()

        return [
            {"name": item["name"], "post_count": item["count"]}
            for item in info["items"]
        ]

    def get_user_info(self, user_ids):
        if len(user_ids) > 100:
            raise ValueError("StackExchange API only supports 100 user ids at a time")

        user_param = self._build_param_list(user_ids)
        # lol this is super gross - I could probably do some url building but this is a one-off scraping script
        url = f"https://api.stackexchange.com/2.2/users/{user_param}?pagesize=100&site=stackoverflow&key={self.config['key']}"
        res = self._make_request("get", url)
        info = res.json()

        return [
            {"id": item["user_id"], "creation_date": item["creation_date"]}
            for item in info["items"]
        ]

    def _build_param_list(self, params):
        return ";".join(str(param) for param in params)

    def _prepare_backoff_if_needed(self, res):
        backoff_seconds = res.get("backoff")
        if not backoff_seconds:
            return

        self.backoff_time = datetime.datetime.now() + datetime.timedelta(
            seconds=backoff_seconds
        )

    def _wait_for_backoff(self):
        if not self.backoff_time:
            return

        time_to_wait = (self.backoff_time - datetime.datetime.now()).total_seconds()
        time.sleep(time_to_wait)
        self.backoff_time = None

    @ratelimit.sleep_and_retry
    @ratelimit.limits(calls=MAX_REQUESTS_PER_SECOND, period=1)
    def _make_request(self, *args, **kwargs):
        self._wait_for_backoff()
        res = self.client.request(*args, **kwargs)
        res.raise_for_status()
        self._prepare_backoff_if_needed(res.json())

        return res


def main():
    # This is test code
    client = Client()
    client.authenticate()
    print(client.get_question_info([10434599, 64734674]))
    print(client.get_tag_info(["python", "go"]))
    print(client.get_user_info([1103734, 1103735]))


if __name__ == "__main__":
    main()
