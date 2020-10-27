import os 
import sys 

import numpy as np 
import pandas as pd 

sys.path.insert(0, './')
from scripts.utils import load_config, check_create_dir

class ModelDataMaker(object):
    """
    Prepare data for model
    """
    def __init__(self, config):
        
        self.config = config 
        self.data_dir = config["data_dir"]
        self.fixture_filepath = os.path.join(self.data_dir, config["file_fixture"])
        self.team_filepath = os.path.join(self.data_dir, config["file_team"])
        self.gw_filepath = os.path.join(self.data_dir, config["file_gw"])
        self.player_filepath = os.path.join(self.data_dir, config["file_player"])

    def get_fixtures_data(self):
        df = pd.read_csv(self.fixture_filepath)
        df.columns = df.columns.str.lower()
        return df

    def get_teams_data(self):
        df = pd.read_csv(self.team_filepath)
        df.columns = df.columns.str.lower()
        return df
    
    def get_gw_data(self):
        try:
            df = pd.read_csv(self.gw_filepath)
        except UnicodeDecodeError: 
            df = pd.read_csv(self.gw_filepath, encoding='latin-1')
        df.columns = df.columns.str.lower()
        return df
    
    def get_players_data(self):
        df = pd.read_csv(self.player_filepath)
        df.columns = df.columns.str.lower()
        return df
    
    def get_player_id_team_id_map(self):
        player_id_team_id_map = {}
        df = self.get_players_data()
        for idx, row in df.iterrows():
            player_id, team_id = int(row["id"]), int(row["team"])
            player_id_team_id_map[player_id] = team_id
        self.player_id_team_id_map = player_id_team_id_map
        return player_id_team_id_map

    def get_player_id_player_name_map(self):
        player_id_player_name_map = {}
        df = self.get_players_data()
        for idx, row in df.iterrows():
            player_id, first_name, second_name = int(row["id"]), row["first_name"], row["second_name"]
            player_name = first_name + " " + second_name
            player_id_player_name_map[player_id] = player_name
        self.player_id_player_name_map = player_id_player_name_map
        return player_id_player_name_map

    def get_player_id_player_position_map(self):
        player_id_player_position_map = {}
        df = self.get_players_data()
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        for idx, row in df.iterrows():
            player_id, element_type = int(row["id"]), int(row["element_type"])
            player_position = position_map.get(element_type, element_type)
            player_id_player_position_map[player_id] = player_position
        self.player_id_player_position_map = player_id_player_position_map
        return player_id_player_position_map
    
    def get_team_id_team_name_map(self):
        team_id_team_name_map = {}
        df = self.get_teams_data()
        df = df.drop_duplicates(subset = ["id"])
        for idx, row in df.iterrows():
            team_id, team_name = int(row["id"]), row["name"]
            team_id_team_name_map[team_id] = team_name
        self.team_id_team_name_map = team_id_team_name_map
        return team_id_team_name_map

    def make_base_data(self):
        
        df_teams = self.get_teams_data()
        df_gw = self.get_gw_data()
        df_players = self.get_players_data()
        df_fixture = self.get_fixtures_data()
        
        df_gw["player_id"] = df_gw["element"].astype(int)
        df_gw["gw_id"] = df_gw["gw"].astype(int)
        df_gw["opp_team_id"] = df_gw["opponent_team"].astype(int)
        df_gw["own_team_id"] = df_gw["player_id"].apply(lambda x: self.player_id_team_id_map[x])
        
        id_cols = ["player_id", "gw_id", "opp_team_id", "own_team_id"]
        remove_cols = ['name', 'kickoff_time', 'gw', 'element', 'opponent_team'] + id_cols
        keep_cols = id_cols + [col for col in df_gw.columns if col not in remove_cols]
        df_gw = df_gw[keep_cols].copy()

        df_teams = df_teams[["id", "name", "strength", "strength_attack_away",
                             "strength_attack_home", "strength_defence_away",
                             "strength_defence_home", "strength_overall_away",
                             "strength_overall_home"]].copy()

        team_cols = list(df_teams.columns)
        df_teams_own = df_teams.copy()
        df_teams_opp = df_teams.copy()
        df_teams_own.columns = ['own_'+ col for col in team_cols]
        df_teams_opp.columns = ['opp_'+ col for col in team_cols]

        df_gw = pd.merge(df_gw, df_teams_own, left_on="own_team_id", right_on="own_id", how="left")
        df_gw = pd.merge(df_gw, df_teams_opp, left_on="opp_team_id", right_on="opp_id", how="left")

        player_id_player_position_map = self.get_player_id_player_position_map() 
        df_gw["element_type"] = df_gw["player_id"].apply(lambda x: player_id_player_position_map[x])
        print(df_gw.head().T)

if __name__ == "__main__":
    config_2020 = {
        "data_dir": "./data/model_data/2020_21/",
        "file_fixture": "fixtures.csv",
        "file_team": "teams.csv",
        "file_gw": "merged_gw.csv",
        "file_player": "players_raw.csv"
    }

    config_2019 = {
        "data_dir": "./data/model_data/2019_20/",
        "file_fixture": "fixtures.csv",
        "file_team": "teams.csv",
        "file_gw": "merged_gw.csv",
        "file_player": "players_raw.csv"
    }

    config_2018 = {
        "data_dir": "./data/model_data/2018_19/",
        "file_fixture": "fixtures.csv",
        "file_team": "teams.csv",
        "file_gw": "merged_gw.csv",
        "file_player": "players_raw.csv"
    }
    import json
    with open(os.path.join(config_2018["data_dir"], 'raw.json'), 'rb') as f:
        data = json.load(f)
    df_teams = pd.DataFrame(data['teams'])
    df_teams.to_csv(os.path.join(config_2018["data_dir"], "teams.csv"), index=False)

    data_maker = ModelDataMaker(config_2018)
    df_fixture = data_maker.get_fixtures_data()
    df_team = data_maker.get_teams_data()
    df_gw = data_maker.get_gw_data()
    df_players = data_maker.get_players_data()

    player_id_team_id_map = data_maker.get_player_id_team_id_map() 
    player_id_player_name_map = data_maker.get_player_id_player_name_map() 
    player_id_player_position_map = data_maker.get_player_id_player_position_map() 
    team_id_team_name_map = data_maker.get_team_id_team_name_map() 

    data_maker.make_base_data()
