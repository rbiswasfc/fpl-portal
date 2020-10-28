import os
import sys
import json
import pickle
import pandas as pd
from tqdm import tqdm

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
        df.columns = df.columns.str.lower()
        df.to_csv(os.path.join(self.data_dir_clean, 'teams.csv'), index=False)
        # print(df.head())
        return df

    def save_players_data(self):
        # data = self.data_scraper.get_all_players_data()
        bootstrap_data = self.data_scraper.get_bootstrap_data()
        data = bootstrap_data['elements']
        file_path = os.path.join(self.data_dir_raw, "players_raw.json")
        save_json_data(data, file_path)
        df = pd.DataFrame(data)
        df.columns = df.columns.str.lower()
        df.to_csv(os.path.join(self.data_dir_clean, 'players_raw.csv'), index=False)
        # print(df.head())
        return df

    def save_fixtures_data(self):
        data = self.data_scraper.get_fixtures_data()
        file_path = os.path.join(self.data_dir_raw, "fixtures.json")
        save_json_data(data, file_path)
        df = pd.DataFrame(data)
        df.columns = df.columns.str.lower()
        df.to_csv(os.path.join(self.data_dir_clean, 'fixtures.csv'), index=False)
        # print(df.head())
        return df

    def save_gameweek_metadata(self):
        data = self.data_scraper.get_gameweek_metadata()
        file_path = os.path.join(self.data_dir_raw, "gw_metadata.json")
        save_json_data(data, file_path)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.data_dir_clean, 'gw_metadata.csv'), index=False)
        # print(df.head())

    def save_gameweek_data(self):
        bootstrap_data = self.data_scraper.get_bootstrap_data()
        this_gw = self.data_scraper.get_next_gameweek_id() - 1
        players = bootstrap_data['elements']
        position_map = {'1': 'GK', '2': 'DEF', '3': 'MID', '4': 'FWD'}
        snapshot_data = []
        player_dfs = []
        for player in tqdm(players):
            player_id = player['id']
            player_name = player['first_name'] + ' ' + player['second_name']
            player_position = position_map[str(int(player['element_type']))]
            team = player['team']
            player['gw'] = this_gw
            player['name'] = player_name
            player['position'] = player_position
            snapshot_data.append(player)
            # print(player)
            player_data = self.data_scraper.get_player_data(player_id)
            df = pd.DataFrame(player_data['history'])
            df['player_id'] = int(player_id)
            df['name'] = player_name
            df['position'] = player_position
            df['team'] = team
            player_dfs.append(df)
            # print(player_id)
        # merge dfs
        snapshot_file_name = "snapshot_players_gw_{}.csv".format(this_gw)
        snapshot_df = pd.DataFrame(snapshot_data)
        snapshot_df.to_csv(os.path.join(self.data_dir_clean, snapshot_file_name), index=False)
        # print(snapshot_df.head())

        df_gws = pd.concat(player_dfs)
        merged_df_path = os.path.join(self.data_dir_clean, "merged_gw_data.csv")
        print(df_gws.head())
        # df_gws.to_csv(merged_df_path, index=False)
        return df_gws

    def save_classic_league_standing(self, league_id="1457340"):
        df_league = self.data_scraper.get_fpl_manager_entry_ids(league_id)
        df_league.to_csv(os.path.join(self.data_dir_clean, "league_{}_standing.csv".format(league_id)),
                         index=False)

    def save_classic_league_history(self, league_id="1457340"):
        df_league = self.data_scraper.get_fpl_manager_entry_ids(league_id)
        manager_ids = df_league["entry_id"].unique().tolist()
        league_history_rows = []
        for manager_id in manager_ids:
            this_manager_history = self.data_scraper.get_entry_data(manager_id)['current']
            for this_gw in this_manager_history:
                this_gw['entry_id'] = manager_id
                league_history_rows.append(this_gw)
        df_league.to_csv(os.path.join(self.data_dir_clean, "league_{}_metadata.csv".format(league_id)))
        df_league_history = pd.DataFrame(league_history_rows)
        df_tmp = df_league[["entry_id", "entry_name", "manager_name"]].copy()
        df_league_history = pd.merge(df_league_history, df_tmp, on='entry_id', how='left')
        df_league_history.to_csv(os.path.join(self.data_dir_clean, "league_{}_history.csv".format(league_id)),
                                 index=False)
        # print(df_league_history.head())

    def save_classic_league_picks(self, league_id="1457340"):

        df_league = self.data_scraper.get_fpl_manager_entry_ids(league_id)
        manager_ids = df_league["entry_id"].unique().tolist()
        this_gw = self.data_scraper.get_next_gameweek_id() - 1
        manage_picks_dict = {}
        for manager_id in manager_ids:
            this_manager_picks = self.data_scraper.get_entry_gw_picks(manager_id, this_gw)
            manage_picks_dict[manager_id] = this_manager_picks

        with open(os.path.join(self.data_dir_clean, "league_{}_manager_picks.pkl".format(league_id)), 'wb') as f:
            pickle.dump(manage_picks_dict, f)


if __name__ == "__main__":
    this_config = {"season": "2020_21", "source_dir": "./data"}
    data_processor = DataProcessor(this_config)
    # data_processor.save_teams_data()
    # data_processor.save_fixtures_data()
    # data_processor.save_players_data()
    # data_processor.save_gameweek_metadata()
    # data_processor.save_gameweek_data()
    # data_processor.save_classic_league_history()
    # data_processor.save_classic_league_picks()
    data_processor.save_classic_league_standing()

    # with open("./data/2020_21/clean/league_1457340_manager_picks.pkl", 'rb') as f:
    #    manage_picks_dict = pickle.load(f)
    # manager_picks = manage_picks_dict[7006192]
    # gw_id = 5
    # for pick in manager_picks:
    #    if pick["entry_history"]["event"] == gw_id:
    #        df_picks = pd.DataFrame(pick["picks"])
    #        print(df_picks)
