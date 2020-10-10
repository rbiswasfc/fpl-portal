import os
import sys

import json
import time
import requests

import numpy as np
import pandas as pd


class DataScraper(object):
    """Fetches data from fantasy premier league api
    """

    def __init__(self):
        self.name = "FPL-Scraper"
        self.data = None

    def get_bootstrap_data(self):
        """Retrieve the fpl player data from static url
        """
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        response = requests.get(url)
        rcode = response.status_code
        if response.status_code != 200:
            raise Exception("Response was code " + str(rcode))
        response_str = response.text
        response = json.loads(response_str)
        self.data = response
        return response


if __name__ == "__main__":
    data_scraper = DataScraper()
    data = data_scraper.get_bootstrap_data()
    print(data.keys())
