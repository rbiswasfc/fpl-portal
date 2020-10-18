import os
import sys
import json
import pandas as pd

sys.path.insert(0, '.')
try:
    from scripts.utils import check_create_dir
    from scripts.data_scrape import DataScraper
except:
    raise ImportError


def save_json_data(data, file_path):
    """
    save data extracted from the FPL api
    :param file_path:
    :type file_path:
    :param data:
    :type data:
    :return:
    :rtype:
    """
    with open(file_path, 'w') as f:
        json.dump(data, f)


class DataProcessor(object):
    """
    Process and save fantasy premier league dat
    """

    def __init__(self, config):

        self.config = config
        self.data_dir = os.path.join(config["source_dir"], config["season"])
        self.data_dir_raw = os.path.join(self.data_dir, 'raw')
        self.data_dir_clean = os.path.join(self.data_dir, 'clean')

        check_create_dir(self.data_dir_raw)
        check_create_dir(self.data_dir_clean)

        self.data_scraper = DataScraper(config)

    def save_teams_data(self):
        data = self.data_scraper.get_team_data()
        file_path = os.path.join(self.data_dir_raw, "teams.json")
        save_json_data(data, file_path)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.data_dir_clean, 'teams.csv'), index=False)
        print(df.head())

    def save_gameweek_data(self):
        pass

    def save_fixtures_data(self):
        pass

    def merge_gameweek_data(self):
        pass


if __name__ == "__main__":
    this_config = {"season": "2020_21", "source_dir": "./data"}
    data_processor = DataProcessor(this_config)
    data_processor.save_teams_data()
