import os
import sys
import pandas as pd


class DataLoader(object):
    def __init__(self, config):
        self.config = config
        self.data_dir = os.path.join(config["source_dir"], config["season"])
        self.data_dir_raw = os.path.join(self.data_dir, 'raw')
        self.data_dir_clean = os.path.join(self.data_dir, 'clean')

    def get_league_standings(self, league_id="1457340"):
        file_path = os.path.join(self.data_dir_clean, "league_{}_standing.csv".format(league_id))
        df = pd.read_csv(file_path)
        return df

    def get_league_gw_history(self, league_id="1457340"):
        file_path = os.path.join(self.data_dir_clean, "league_{}_history.csv".format(league_id))
        df = pd.read_csv(file_path)
        return df
