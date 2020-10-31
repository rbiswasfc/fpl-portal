import os
import sys
import time
import pickle
import pandas as pd

sys.path.insert(0, '.')
try:
    from scripts.utils import check_create_dir
    from scripts.data_scrape import DataScraper
    from scripts.data_processor import DataProcessor
except:
    raise ImportError


def check_cache_validity(file_path, valid_days=0.05):
    if os.path.isfile(file_path):
        mod_time = os.path.getmtime(file_path)
        if (time.time() - mod_time) / 3600 < 24 * valid_days:
            return True
    return False


class DataLoader(object):
    def __init__(self, config):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.data_dir = os.path.join(config["source_dir"], config["season"])
        self.data_dir_raw = os.path.join(self.data_dir, 'raw')
        self.data_dir_clean = os.path.join(self.data_dir, 'clean')
        self.current_gw = int(self.data_processor.data_scraper.get_next_gameweek_id() - 1)

    def get_league_standings(self, league_id="1457340"):
        file_path = os.path.join(self.data_dir_clean, "league_{}_standing.csv".format(league_id))
        if check_cache_validity(file_path):
            print("Valid cache found for {}".format(file_path))
        else:
            self.data_processor.save_classic_league_standing(league_id)
        df = pd.read_csv(file_path)
        return df

    def get_league_gw_history(self, league_id="1457340"):
        file_path = os.path.join(self.data_dir_clean, "league_{}_history.csv".format(league_id))
        if check_cache_validity(file_path):
            print("Valid cache found for {}".format(file_path))
        else:
            self.data_processor.save_classic_league_history(league_id)
        df = pd.read_csv(file_path)
        return df

    def get_league_picks_history(self, league_id="1457340"):
        file_path = os.path.join(self.data_dir_clean, "league_{}_manager_picks_history.pkl".format(league_id))
        if check_cache_validity(file_path):
            print("Valid cache found for {}".format(file_path))
        else:
            self.data_processor.save_classic_league_picks_history(league_id)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def get_manager_current_gw_picks(self, manager_id):
        file_path = os.path.join(self.data_dir_clean, "manager_{}_picks_gw_{}.pkl".format(manager_id, self.current_gw))
        if check_cache_validity(file_path):
            print("Valid cache found for {}".format(file_path))
        else:
            self.data_processor.save_manager_current_gw_picks(manager_id)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
