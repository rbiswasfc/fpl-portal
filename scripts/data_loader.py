import os
import sys
import pickle
import pandas as pd

sys.path.insert(0, '.')
try:
    from scripts.utils import check_create_dir
    from scripts.data_scrape import DataScraper
    from scripts.data_processor import DataProcessor
    from scripts.utils import check_cache_validity
except:
    raise ImportError


class DataLoader(object):
    """
    Interface to load data as need for FPL models
    """
    def __init__(self, config):
        """
        initialization of data loader class
        :param config: config file
        :type config: dict
        """
        self.config = config
        self.data_processor = DataProcessor(config)
        self.data_dir = os.path.join(config["source_dir"], config["season"])
        self.data_dir_raw = os.path.join(self.data_dir, 'raw')
        self.data_dir_clean = os.path.join(self.data_dir, 'clean')
        self.current_gw = int(self.data_processor.data_scraper.get_next_gameweek_id() - 1)

    def get_league_standings(self, league_id="1457340"):
        """
        Load current league standings
        :param league_id: league code
        :type league_id: str
        :return league standing df
        :rtype pd.DataFrame
        """
        file_path = os.path.join(self.data_dir_clean, "league_{}_standing.csv".format(league_id))
        if check_cache_validity(file_path):
            print("Valid cache found for {}".format(file_path))
        else:
            self.data_processor.save_classic_league_standing(league_id)
        df = pd.read_csv(file_path)
        return df

    def get_league_gw_history(self, league_id="1457340"):
        """
        Load league gameweek points history
        :param league_id: league code
        :type league_id: str
        :return: league points history df
        :rtype: pd.DataFrame
        """
        file_path = os.path.join(self.data_dir_clean, "league_{}_history.csv".format(league_id))
        if check_cache_validity(file_path):
            print("Valid cache found for {}".format(file_path))
        else:
            self.data_processor.save_classic_league_history(league_id)
        df = pd.read_csv(file_path)
        return df

    def get_league_picks_history(self, league_id="1457340"):
        """
        Load teams picks history for a given league
        :param league_id: league code
        :type league_id: str
        :return: data with league picks
        :rtype: dict
        """
        file_path = os.path.join(self.data_dir_clean, "league_{}_manager_picks_history.pkl".format(league_id))
        if check_cache_validity(file_path):
            print("Valid cache found for {}".format(file_path))
        else:
            self.data_processor.save_classic_league_picks_history(league_id)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def get_manager_current_gw_picks(self, manager_id):
        """
        Load a manager's current gameweek team picks
        :param manager_id: manager id
        :type manager_id: str
        :return: managers team picks data
        :rtype: List
        """
        file_path = os.path.join(self.data_dir_clean, "manager_{}_picks_gw_{}.pkl".format(manager_id, self.current_gw))
        if check_cache_validity(file_path):
            print("Valid cache found for {}".format(file_path))
        else:
            self.data_processor.save_manager_current_gw_picks(manager_id)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def get_next_gameweek_id(self):
        """
        Load next gameweek id
        :return: next gameweek id
        :rtype: int
        """
        file_path = os.path.join(self.data_dir_clean, "next_gw_id.pkl")
        if check_cache_validity(file_path):
            print("Valid cache found for {}".format(file_path))
        else:
            print("executing scrape ...")
            self.data_processor.save_next_gw_id()
        with open(file_path, 'rb') as f:
            gw = pickle.load(f)
        return gw

    def get_live_gameweek_data(self):
        """
        Load live gameweek points data
        :return: live gameweek df
        :rtype: pd.DataFrame
        """
        this_gw = self.get_next_gameweek_id() - 1
        file_path = os.path.join(self.data_dir_clean, "snapshot_players_gw_{}.csv".format(this_gw))
        if check_cache_validity(file_path):
            print("Valid cache found for {}".format(file_path))
        else:
            print("executing scrape ...")
            self.data_processor.save_gameweek_data()
        df = pd.read_csv(file_path)
        return df

    def get_top_manager_picks(self):
        """
        Load top manager picks data for the current gameweek
        :return: top managers picks dataframe
        :rtype: pd.DataFrame()
        """
        this_gw = int(self.get_next_gameweek_id() - 1)
        file_path = os.path.join(self.data_dir_clean, "top_manager_picks_gw_{}.csv".format(this_gw))
        if check_cache_validity(file_path, valid_days=5.0):
            print("Valid cache found for {}".format(file_path))
        else:
            print("executing scrape ...")
            self.data_processor.save_top_manager_picks(n_pages=12)
        df = pd.read_csv(file_path)
        return df

    def get_manager_bank_balance(self, manager_id):
        """
        Load a manager's bank balance
        :param manager_id: manager id
        :type manager_id: str
        :return: bank balance
        :rtype: float
        """
        this_gw = int(self.get_next_gameweek_id() - 1)
        file_path = os.path.join(self.data_dir_clean, "manager_{}_bank_gw_{}.csv".format(manager_id, this_gw))
        if check_cache_validity(file_path):
            print("Valid cache found for {}".format(file_path))
        else:
            print("executing scrape ...")
            self.data_processor.save_manager_bank_balance(manager_id)
        with open(file_path, 'rb') as f:
            bb = pickle.load(f)
        return bb

    def get_gameweek_metadata(self):
        """
        Load gameweek metadata
        :return:
        :rtype:
        """
        file_path = os.path.join(self.data_dir_clean, 'gw_metadata.csv')
        if check_cache_validity(file_path):
            print("Valid cache found for {}".format(file_path))
        else:
            print("executing scrape ...")
            self.data_processor.save_gameweek_metadata()
        df = pd.read_csv(file_path)
        return df


if __name__ == "__main__":
    pass
