import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
import pdb
from datetime import datetime, timedelta

sys.path.insert(0, './')

try:
    from scripts.utils import load_config, check_create_dir
    from scripts.data_scrape import DataScraper
except:
    raise ImportError


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
        self.understat_team_filepath = os.path.join(self.data_dir, config["file_understat_team"])

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

    def get_understat_team_data(self):
        with open(self.understat_team_filepath, 'rb') as f:
            data = pickle.load(f)

        # all_team_ids = list(data.keys())
        # names = []
        # for team_id in all_team_ids:
        #    names.append(data[team_id]['title'])
        # for name in sorted(names):
        #    print(name)
        return data

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
        df = df.drop_duplicates(subset=["id"])
        for idx, row in df.iterrows():
            team_id, team_name = int(row["id"]), row["name"]
            team_id_team_name_map[team_id] = team_name
        self.team_id_team_name_map = team_id_team_name_map
        return team_id_team_name_map

    def get_team_name_team_id_map(self):
        team_id_team_name_map = self.get_team_id_team_name_map()
        team_name_team_id_map = {}
        for k, v in team_id_team_name_map.items():
            team_name_team_id_map[v] = k
        self.team_name_team_id_map = team_name_team_id_map
        return team_name_team_id_map

    def get_player_id_cost_map(self):
        player_id_cost_map = {}
        df = self.get_players_data()
        for idx, row in df.iterrows():
            player_id, now_cost = int(row["id"]), row["now_cost"]
            player_id_cost_map[player_id] = now_cost
        self.player_id_cost_map = player_id_cost_map
        return player_id_cost_map

    def get_player_id_play_chance_map(self):
        player_id_play_chance_map = {}
        df = self.get_players_data()
        for idx, row in df.iterrows():
            player_id, chance = int(row["id"]), row["chance_of_playing_next_round"]
            player_id_play_chance_map[player_id] = chance
        self.player_id_play_chance_map = player_id_play_chance_map
        return player_id_play_chance_map

    def get_player_id_selection_map(self):
        player_id_selection_map = {}
        df = self.get_players_data()
        for idx, row in df.iterrows():
            player_id, pct = int(row["id"]), row["selected_by_percent"]
            player_id_selection_map[player_id] = pct
        self.player_id_selection_map = player_id_selection_map
        return player_id_selection_map

    def get_player_id_ave_points_map(self):
        player_id_ave_points_map = {}
        df = self.get_players_data()
        for idx, row in df.iterrows():
            player_id, ave_pts = int(row["id"]), row["points_per_game"]
            player_id_ave_points_map[player_id] = ave_pts
        self.player_id_ave_points_map = player_id_ave_points_map
        return player_id_ave_points_map

    def prepare_understat_data(self):
        data = self.get_understat_team_data()

        with open(os.path.join(self.data_dir, "understat_team_mapping.json"), 'rb') as f:
            team_name_mapping = json.load(f)

        team_name_team_id_map = self.get_team_name_team_id_map()
        understat_ids = list(data.keys())
        dfs = []
        for this_id in understat_ids:
            this_data = data[this_id]

            team_name = this_data['title']
            team_history_data = this_data['history']
            team_df = pd.DataFrame(team_history_data)
            # print(team_df.columns)
            team_df['date'] = pd.to_datetime(team_df['date'])
            team_df = team_df.sort_values(by='date')
            team_df["effective_gw_id"] = [i + 1 for i in range(len(team_df))]
            team_df["ppda_att"] = team_df["ppda"].apply(lambda x: x['att'])
            team_df["ppda_def"] = team_df["ppda"].apply(lambda x: x['def'])
            team_df["ppda_allowed_att"] = team_df["ppda_allowed"].apply(lambda x: x['att'])
            team_df["ppda_allowed_def"] = team_df["ppda_allowed"].apply(lambda x: x['def'])
            team_df = team_df.drop(columns=["ppda", "ppda_allowed"])
            team_df["team_name"] = team_name
            team_df["fpl_team_name"] = team_df["team_name"].apply(lambda x: team_name_mapping.get(x, x))
            team_df["understat_id"] = this_id
            team_df["team_id"] = team_df["fpl_team_name"].apply(lambda x: team_name_team_id_map.get(x, 'NA'))
            dfs.append(team_df)
        df_understat = pd.concat(dfs)
        df_understat.columns = df_understat.columns.str.lower()
        return df_understat

    def get_effective_gameweek_map(self):
        df_fixtures = self.get_fixtures_data()

        all_teams = df_fixtures["team_a"].unique().tolist()
        dfs = []
        for this_team in all_teams:
            df_this_team = df_fixtures[
                (df_fixtures["team_a"] == this_team) | (df_fixtures["team_h"] == this_team)].copy()
            df_this_team["kickoff_time"] = pd.to_datetime(df_this_team["kickoff_time"])
            last_kickoff = df_this_team["kickoff_time"].max()
            df_this_team["kickoff_time"] = df_this_team["kickoff_time"].fillna(last_kickoff + timedelta(days=365))
            df_this_team = df_this_team.sort_values(by="kickoff_time")

            df_this_team = df_this_team.rename(columns={"event": "gw_id"})
            df_this_team["gw_id"] = df_this_team["gw_id"].fillna(-1)
            df_this_team["gw_id"] = df_this_team["gw_id"].astype(int)
            df_this_team["own_team_id"] = int(this_team)
            df_this_team["fixture_opp_team_id"] = df_this_team[["team_h", "team_a"]].apply(
                lambda x: x[0] if x[0] != this_team else x[1], axis=1)
            df_this_team["home_flag"] = df_this_team[["team_h", "team_a"]].apply(
                lambda x: True if x[0] == this_team else False, axis=1)
            df_this_team["effective_gw_id"] = [i + 1 for i in range(len(df_this_team))]
            df_this_team = df_this_team[
                ["own_team_id", "gw_id", "effective_gw_id", "fixture_opp_team_id", "home_flag"]].copy()
            dfs.append(df_this_team)
        df_map = pd.concat(dfs)
        return df_map

    def make_nth_gw_scoring_base(self, gw):
        player_id_team_id_map = self.get_player_id_team_id_map()
        all_players = list(player_id_team_id_map.keys())
        df_map = self.get_effective_gameweek_map()

        df = pd.DataFrame()
        df["player_id"] = all_players
        df["own_team_id"] = df["player_id"].apply(lambda x: player_id_team_id_map.get(x, -1))
        df["gw_id"] = gw
        df = pd.merge(df, df_map, how="left", on=["own_team_id", "gw_id"])
        df["opp_team_id"] = df["fixture_opp_team_id"]
        df = df.drop_duplicates(subset=["player_id", "gw_id"])
        df = df[["player_id", "gw_id", "opp_team_id"]].copy()

        df_teams = self.get_teams_data()
        df_teams = df_teams[["id", "name", "strength", "strength_attack_away",
                        "strength_attack_home", "strength_defence_away",
                        "strength_defence_home", "strength_overall_away",
                        "strength_overall_home"]].copy()
        team_cols = list(df_teams.columns)
        df_teams_opp = df_teams.copy()
        df_teams_opp.columns = ['opp_' + col for col in team_cols]
        df = pd.merge(df, df_teams_opp, how='left', left_on="opp_team_id", right_on="opp_id")
        keep_cols = ["player_id", "gw_id"] + ['opp_' + col for col in team_cols]
        df = df[keep_cols].copy()
        return df

    def make_scoring_base(self):
        player_id_team_id_map = self.get_player_id_team_id_map()
        all_players = list(player_id_team_id_map.keys())
        df_map = self.get_effective_gameweek_map()
        try:
            scoring_gw = int(self.config["scoring_gw"])
        except:
            print("No Valid Scoring GW provided")
            return pd.DataFrame()
        print("=="*20)
        print("getting scoring df...")
        df_scoring = pd.DataFrame()
        df_scoring["player_id"] = all_players
        df_scoring["own_team_id"] = df_scoring["player_id"].apply(lambda x: player_id_team_id_map.get(x, -1))
        df_scoring["gw_id"] = scoring_gw
        df_scoring = pd.merge(df_scoring, df_map, how="left", on=["own_team_id", "gw_id"])
        df_scoring["opp_team_id"] = df_scoring["fixture_opp_team_id"]
        df_scoring["is_home"] = df_scoring["home_flag"]
        # player_id_player_position_map = self.get_player_id_player_position_map()
        # df_scoring["element_type"] = df_scoring["player_id"].apply(lambda x: player_id_player_position_map[x])
        print("shape of scoring df: {}".format(df_scoring.shape))
        return df_scoring

    def make_base_data(self):
        df_teams = self.get_teams_data()
        df_gw = self.get_gw_data()
        df_players = self.get_players_data()
        df_fixture = self.get_fixtures_data()

        df_gw["player_id"] = df_gw["element"].astype(int)
        df_gw["gw_id"] = df_gw["gw"].astype(int)
        df_gw["opp_team_id"] = df_gw["opponent_team"].astype(int)
        player_id_team_id_map = self.get_player_id_team_id_map()
        df_gw["own_team_id"] = df_gw["player_id"].apply(lambda x: player_id_team_id_map[x])
        df_gw["is_home"] = df_gw["was_home"]

        id_cols = ["player_id", "gw_id", "opp_team_id", "own_team_id"]
        remove_cols = ['name', 'kickoff_time', 'gw', 'element', 'opponent_team'] + id_cols
        keep_cols = id_cols + [col for col in df_gw.columns if col not in remove_cols]
        df_gw = df_gw[keep_cols].copy()

        # get effective GW map
        df_map = self.get_effective_gameweek_map()
        # df_gw = pd.merge(df_gw, df_map, how='left', on=['own_team_id', 'gw_id'])
        df_gw = pd.merge(df_gw, df_map, how='left', left_on=['own_team_id', 'opp_team_id', 'gw_id'],
                         right_on=['own_team_id', 'fixture_opp_team_id', 'gw_id'])

        # concat scoring dataframe
        df_scoring = self.make_scoring_base()
        if len(df_scoring) > 0:
            df_gw = pd.concat([df_gw, df_scoring])

        df_teams = df_teams[["id", "name", "strength", "strength_attack_away",
                             "strength_attack_home", "strength_defence_away",
                             "strength_defence_home", "strength_overall_away",
                             "strength_overall_home"]].copy()

        team_cols = list(df_teams.columns)
        df_teams_own = df_teams.copy()
        df_teams_opp = df_teams.copy()
        df_teams_own.columns = ['own_' + col for col in team_cols]
        df_teams_opp.columns = ['opp_' + col for col in team_cols]

        df_gw = pd.merge(df_gw, df_teams_own, left_on="own_team_id", right_on="own_id", how="left")
        df_gw = pd.merge(df_gw, df_teams_opp, left_on="opp_team_id", right_on="opp_id", how="left")

        player_id_player_position_map = self.get_player_id_player_position_map()
        df_gw["position"] = df_gw["player_id"].apply(lambda x: player_id_player_position_map[x])

        def get_future_points(n_future):
            df_tmp = df_gw[["player_id", "effective_gw_id", "total_points"]].copy()
            new_col = "total_points_next_{}".format(n_future)
            df_tmp["effective_gw_id"] = df_tmp["effective_gw_id"] - n_future
            df_tmp = df_tmp.rename(columns={"total_points":new_col})
            df_tmp = df_tmp.drop_duplicates(subset=["player_id", "effective_gw_id"])
            return df_tmp
        df_next_1 = get_future_points(1)
        df_next_2 = get_future_points(2)
        df_gw = pd.merge(df_gw, df_next_1, how='left', on=["player_id", "effective_gw_id"])
        df_gw = pd.merge(df_gw, df_next_2, how='left', on=["player_id", "effective_gw_id"])
        df_gw["potential"] = df_gw["total_points"] + df_gw["total_points_next_1"] + df_gw["total_points_next_2"] 

        # print(df_gw.head().T)
        # print(df_gw.tail().T)

        return df_gw


if __name__ == "__main__":
    scraper_config = {"season": "2020_21", "source_dir": "./data/raw/"}
    data_scraper = DataScraper(scraper_config)
    scoring_gw = data_scraper.get_next_gameweek_id()
    config_2020 = {
        "data_dir": "./data/model_data/2020_21/",
        "file_fixture": "fixtures.csv",
        "file_team": "teams.csv",
        "file_gw": "merged_gw.csv",
        "file_player": "players_raw.csv",
        "file_understat_team": "understat_team_data.pkl",
        "scoring_gw": scoring_gw
    }

    config_2019 = {
        "data_dir": "./data/model_data/2019_20/",
        "file_fixture": "fixtures.csv",
        "file_team": "teams.csv",
        "file_gw": "merged_gw.csv",
        "file_player": "players_raw.csv",
        "file_understat_team": "understat_team_data.pkl",
        "scoring_gw": "NA"
    }

    config_2018 = {
        "data_dir": "./data/model_data/2018_19/",
        "file_fixture": "fixtures.csv",
        "file_team": "teams.csv",
        "file_gw": "merged_gw.csv",
        "file_player": "players_raw.csv",
        "file_understat_team": "understat_team_data.pkl",
        "scoring_gw": "NA"
    }
    # import json

    # with open(os.path.join(config_2018["data_dir"], 'raw.json'), 'rb') as f:
    #    data = json.load(f)
    # df_teams = pd.DataFrame(data['teams'])
    # df_teams.to_csv(os.path.join(config_2018["data_dir"], "teams.csv"), index=False)

    data_maker = ModelDataMaker(config_2020)
    # df_fixture = data_maker.get_fixtures_data()
    # df_team = data_maker.get_teams_data()
    # df_gw = data_maker.get_gw_data()
    # df_players = data_maker.get_players_data()

    # player_id_team_id_map = data_maker.get_player_id_team_id_map() 
    # player_id_player_name_map = data_maker.get_player_id_player_name_map() 
    # player_id_player_position_map = data_maker.get_player_id_player_position_map() 
    # team_id_team_name_map = data_maker.get_team_id_team_name_map() 

    df = data_maker.make_base_data()
    # print(df.sample(5))
    # df_understat_tmp = data_maker.prepare_understat_data()
    # print(df_understat_tmp.sample(5).T)

    # df_map_tmp = data_maker.get_effective_gameweek_map()
    # print(df_map_tmp.sample(10))

    data_maker.make_scoring_base()

    # pdb.set_trace()
