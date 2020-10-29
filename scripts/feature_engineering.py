import os
import sys
import json
import pickle
import pdb
import numpy as np
import pandas as pd
import pdb
from datetime import datetime, timedelta

sys.path.insert(0, './')

try:
    from scripts.utils import load_config, check_create_dir
    from scripts.data_preparation import ModelDataMaker
    from scripts.data_scrape import DataScraper
except:
    raise ImportError


class FeatureEngineering(object):
    def __init__(self, scoring=False):
        self.lag_list = [1, 2, 3]
        self.next_list = [1, 2]
        self.feature_dict = dict()
        self.feature_dict["features"] = []
        self.feature_dict["num_features"] = []
        self.feature_dict["cat_features"] = []
        self.scoring = scoring

    def make_lag_features(self, df, index_cols, shift_col, feat_list, feat_type):
        added_dfs = []
        for feat in feat_list:
            for lag in self.lag_list:
                new_feat = "{}_lag_{}".format(feat, lag)
                self.feature_dict["features"].append(new_feat)
                if feat_type == 'cat':
                    self.feature_dict["cat_features"].append(new_feat)
                else:
                    self.feature_dict["num_features"].append(new_feat)
                focus_cols = index_cols + [feat]
                df_focus = df[focus_cols].copy()
                df_focus[shift_col] = df_focus[shift_col].apply(lambda x: int(x + lag))
                df_focus[new_feat] = df_focus[feat]
                df_focus.drop(columns=[feat], axis=1, inplace=True)
                added_dfs.append(df_focus)
        return added_dfs

    def make_next_features(self, df, index_cols, shift_col, feat_list, feat_type):
        added_dfs = []
        for feat in feat_list:
            for next_num in self.next_list:
                new_feat = "{}_next_{}".format(feat, next_num)
                self.feature_dict["features"].append(new_feat)
                if feat_type == 'cat':
                    self.feature_dict["cat_features"].append(new_feat)
                else:
                    self.feature_dict["num_features"].append(new_feat)
                focus_cols = index_cols + [feat]
                df_focus = df[focus_cols].copy()
                df_focus[shift_col] = df_focus[shift_col].apply(lambda x: int(x - next_num))
                df_focus[new_feat] = df_focus[feat]
                df_focus.drop(columns=[feat], axis=1, inplace=True)
                added_dfs.append(df_focus)
        return added_dfs

    def execute_fe(self, config):
        gw_cat_features = ["was_home", "team_h_score", "team_a_score", "goals_scored", "assists",
                           "clean_sheets", "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
                           "yellow_cards", "red_cards", "saves", "bonus"]
        gw_num_features = ["minutes", "bps", "influence", "creativity", "threat", "ict_index", "selected",
                           "transfers_in", "transfers_out", "transfers_balance"]
        team_cat_features = ["strength"]
        team_num_features = ["strength_overall_home", "strength_overall_away", "strength_attack_home",
                             "strength_attack_away", "strength_defence_home", "strength_defence_away"]
        understat_cat_features = []
        understat_num_features = ["xg", "xga", "npxg", "npxga", "deep", "deep_allowed", "xpts", "npxgd",
                                  "ppda_att", "ppda_def", "ppda_allowed_att", "ppda_allowed_def"]

        static_cat_features = ["position", "is_home"]  # mis-leading name? these are non-lag feats
        static_num_features = []

        own_team_cat_features = ["own_" + feat for feat in team_cat_features]
        own_team_num_features = ["own_" + feat for feat in team_num_features]
        opp_team_cat_features = ["opp_" + feat for feat in team_cat_features]
        opp_team_num_features = ["opp_" + feat for feat in team_num_features]

        static_cat_features = static_cat_features + own_team_cat_features + opp_team_cat_features
        static_num_features = static_num_features + own_team_num_features + opp_team_num_features

        self.feature_dict["features"].extend(static_cat_features)
        self.feature_dict["features"].extend(static_num_features)
        self.feature_dict["cat_features"].extend(static_cat_features)
        self.feature_dict["num_features"].extend(static_num_features)

        data_maker = ModelDataMaker(config)
        df_base = data_maker.make_base_data()
        df_understat = data_maker.prepare_understat_data()

        print("Removing invalid data points from GW Data...")
        print("Shape before removal: {}".format(df_base.shape))
        df_base = df_base[~df_base["effective_gw_id"].isna()].copy()
        df_base = df_base[df_base["effective_gw_id"] > 0].copy()
        print("Shape after removal: {}".format(df_base.shape))

        df_base["player_id"] = df_base["player_id"].astype(int)
        df_base["effective_gw_id"] = df_base["effective_gw_id"].astype(int)
        df_base["own_team_id"] = df_base["own_team_id"].astype(int)
        df_base["opp_team_id"] = df_base["opp_team_id"].astype(int)

        df_understat_own = df_understat.copy()
        df_understat_opp = df_understat.copy()
        understat_focus_cols = understat_cat_features + understat_num_features
        understat_own_col_map, understat_opp_col_map = dict(), dict()
        for col in understat_focus_cols:
            understat_own_col_map[col] = "own_" + col
            understat_opp_col_map[col] = "opp_" + col

        df_understat_own = df_understat_own.rename(columns=understat_own_col_map)
        df_understat_opp = df_understat_opp.rename(columns=understat_opp_col_map)

        opp_cat_lag_dfs = self.make_lag_features(df_base, ["player_id", "effective_gw_id"],
                                                 "effective_gw_id", opp_team_cat_features, 'cat')
        opp_num_lag_dfs = self.make_lag_features(df_base, ["player_id", "effective_gw_id"],
                                                 "effective_gw_id", opp_team_num_features, 'num')

        gw_cat_lag_dfs = self.make_lag_features(df_base, ["player_id", "effective_gw_id"],
                                                "effective_gw_id", gw_cat_features, 'cat')
        gw_num_lag_dfs = self.make_lag_features(df_base, ["player_id", "effective_gw_id"],
                                                "effective_gw_id", gw_num_features, 'num')

        understat_own_cat_features = ["own_" + feat for feat in understat_cat_features]
        understat_own_num_features = ["own_" + feat for feat in understat_num_features]
        understat_own_cat_lag_dfs = self.make_lag_features(df_understat_own, ["team_id", "effective_gw_id"],
                                                           "effective_gw_id", understat_own_cat_features, 'cat')
        understat_own_num_lag_dfs = self.make_lag_features(df_understat_own, ["team_id", "effective_gw_id"],
                                                           "effective_gw_id", understat_own_num_features, 'num')

        understat_opp_cat_features = ["opp_" + feat for feat in understat_cat_features]
        understat_opp_num_features = ["opp_" + feat for feat in understat_num_features]
        understat_opp_cat_lag_dfs = self.make_lag_features(df_understat_opp, ["team_id", "effective_gw_id"],
                                                           "effective_gw_id", understat_opp_cat_features, 'cat')
        understat_opp_num_lag_dfs = self.make_lag_features(df_understat_opp, ["team_id", "effective_gw_id"],
                                                           "effective_gw_id", understat_opp_num_features, 'num')

        print("Merging Gameweek Cat Features")
        print("Shape before merge: {}".format(df_base.shape))
        for df in gw_cat_lag_dfs:
            df = df.drop_duplicates(subset=["player_id", "effective_gw_id"]).copy()
            df_base = pd.merge(df_base, df, how="left", on=["player_id", "effective_gw_id"])
        print("Shape after merge: {}".format(df_base.shape))

        print("Merging Gameweek Num Features")
        print("Shape before merge: {}".format(df_base.shape))
        for df in gw_num_lag_dfs:
            df = df.drop_duplicates(subset=["player_id", "effective_gw_id"]).copy()
            df_base = pd.merge(df_base, df, how="left", on=["player_id", "effective_gw_id"])
        print("Shape after merge: {}".format(df_base.shape))

        print("Merging Opp Cat Team Features")
        print("Shape before merge: {}".format(df_base.shape))
        for df in opp_cat_lag_dfs:
            df = df.drop_duplicates(subset=["player_id", "effective_gw_id"]).copy()
            df_base = pd.merge(df_base, df, how="left", on=["player_id", "effective_gw_id"])
        print("Shape after merge: {}".format(df_base.shape))

        print("Merging Opp Num Team Features")
        print("Shape before merge: {}".format(df_base.shape))
        for df in opp_num_lag_dfs:
            df = df.drop_duplicates(subset=["player_id", "effective_gw_id"]).copy()
            df_base = pd.merge(df_base, df, how="left", on=["player_id", "effective_gw_id"])
        print("Shape after merge: {}".format(df_base.shape))

        # add understat data own team
        print("Merging understat own team data")
        for df in understat_own_num_lag_dfs:
            df = df.drop_duplicates(subset=["team_id", "effective_gw_id"]).copy()
            df = df.rename(columns={"team_id": "own_team_id"})
            df_base = pd.merge(df_base, df, how="left", on=["own_team_id", "effective_gw_id"])
        print("Shape after merge: {}".format(df_base.shape))

        # add understat data opp team
        print("Merging understat opp team data")
        for df in understat_opp_num_lag_dfs:
            df = df.drop_duplicates(subset=["team_id", "effective_gw_id"]).copy()
            df = df.rename(columns={"team_id": "opp_team_id"})
            df_base = pd.merge(df_base, df, how="left", on=["opp_team_id", "effective_gw_id"])
        print("Shape after merge: {}".format(df_base.shape))

        return df_base


def make_XY_data(dataset_dir="./data/model_data/xy_data/"):
    # configs
    check_create_dir(dataset_dir)
    scraper_config = {"season": "2020_21", "source_dir": "./data/raw/"}
    data_scraper = DataScraper(scraper_config)
    scoring_gw = data_scraper.get_next_gameweek_id()

    fe_2020 = FeatureEngineering()

    config_2020 = {
        "data_dir": "./data/model_data/2020_21/",
        "file_fixture": "fixtures.csv",
        "file_team": "teams.csv",
        "file_gw": "merged_gw.csv",
        "file_player": "players_raw.csv",
        "file_understat_team": "understat_team_data.pkl",
        "scoring_gw": scoring_gw
    }

    df_2020 = fe_2020.execute_fe(config_2020)

    fe_2019 = FeatureEngineering()
    config_2019 = {
        "data_dir": "./data/model_data/2019_20/",
        "file_fixture": "fixtures.csv",
        "file_team": "teams.csv",
        "file_gw": "merged_gw.csv",
        "file_player": "players_raw.csv",
        "file_understat_team": "understat_team_data.pkl",
        "scoring_gw": "NA"
    }
    df_2019 = fe_2019.execute_fe(config_2019)

    fe_2018 = FeatureEngineering()
    config_2018 = {
        "data_dir": "./data/model_data/2018_19/",
        "file_fixture": "fixtures.csv",
        "file_team": "teams.csv",
        "file_gw": "merged_gw.csv",
        "file_player": "players_raw.csv",
        "file_understat_team": "understat_team_data.pkl",
        "scoring_gw": "NA"
    }
    df_2018 = fe_2018.execute_fe(config_2018)

    df_2018["season_id"] = 0
    df_2019["season_id"] = 1
    df_2020["season_id"] = 2

    df_XY = pd.concat([df_2018, df_2019, df_2020])
    df_XY["global_gw_id"] = df_XY[["season_id", "gw_id"]].apply(lambda x: x[0] * 100 + x[1], axis=1)

    # FIX: was home
    df_XY["was_home_lag_1"] = df_XY["was_home_lag_1"].astype(bool)
    df_XY["was_home_lag_2"] = df_XY["was_home_lag_2"].astype(bool)
    df_XY["was_home_lag_3"] = df_XY["was_home_lag_3"].astype(bool)

    # cat cols
    features_dict = fe_2020.feature_dict
    cat_features = features_dict["cat_features"]

    #
    cat_list = []
    type_dict = dict(df_XY.dtypes)
    for k, v in type_dict.items():
        if str(v) == 'object':
            cat_list.append(k)

    # print(cat_list)

    for feat in cat_features:
        if feat in cat_list:
            # print(feat)
            df_XY[feat] = df_XY[feat].astype('category').cat.codes

    pts_clip = 10
    star_clip = 5
    df_XY["reg_target"] = df_XY["total_points"].apply(lambda x: x if x <= pts_clip else pts_clip)
    df_XY["star_target"] = df_XY["total_points"].apply(lambda x: 1 if x >= star_clip else 0)

    df_XY["global_gw_id"] = df_XY["global_gw_id"].fillna(-1)
    df_XY["global_gw_id"] = df_XY["global_gw_id"].astype(int)
    global_scoring_gw = df_XY["global_gw_id"].max()
    global_test_gw = global_scoring_gw - 1
    df_XY_train = df_XY[df_XY["global_gw_id"] < global_test_gw].copy()
    df_XY_test = df_XY[df_XY["global_gw_id"] == global_test_gw].copy()
    df_XY_scoring = df_XY[df_XY["global_gw_id"] == global_scoring_gw].copy()

    # save XY data
    df_XY_train.to_csv(os.path.join(dataset_dir, "xy_train.csv"), index=False)
    df_XY_test.to_csv(os.path.join(dataset_dir, "xy_test.csv"), index=False)
    df_XY_scoring.to_csv(os.path.join(dataset_dir, "xy_scoring.csv"), index=False)

    with open(os.path.join(dataset_dir, "features_after_fe.pkl"), 'wb') as f:
        pickle.dump(features_dict, f)

    # pdb.set_trace()


if __name__ == "__main__":
    make_XY_data()
