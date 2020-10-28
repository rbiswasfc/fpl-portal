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
except:
    raise ImportError


class FeatureEngineering(object):
    def __init__(self):
        self.lag_list = [1, 2, 3]
        self.next_list = [1, 2]
        self.feature_dict = dict()
        self.feature_dict["features"] = []
        self.feature_dict["num_features"] = []
        self.feature_dict["cat_features"] = []

    def make_gw_lag_features(self, df, index_cols, feature_list):
        added_dfs = [], []
        for feat in feature_list:
            for lag in self.lag_list:
                new_feat = "{}_lag_{}".format(feat, lag)
                added_feats.append(new_feat)

    def make_understat_lag_features(self):
        pass

    def make_next_features(self):
        pass

    def execute_fe(self):
        gw_cat_features = ["was_home", "opponent_team", "team_h_score", "team_a_score", "goals_scored", "assists",
                           "clean_sheets", "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
                           "yellow_cards", "red_cards", "saves", "bonus", "position"]
        gw_num_features = ["minutes", "bps", "influence", "creativity", "threat", "ict_index", "selected",
                           "transfer_in", "transfer_out", "transfer_balance"]
        teams_cat_features = ["strength", "win", "loss"]
        teams_num_features = ["strength_overall_home", "strength_overall_away", "strength_attack_home",
                              "strength_attack_away", "strength_defence_home", "strength_defence_away"]
        understat_cat_features = ["pts"]
        understat_num_features = ["xg", "xga", "npxg", "npxga", "deep", "deep_allowed", "xpts", "nxpgd",
                                  "ppda_att", "ppda_def", "ppda_allowed_att", "ppda_allowed_def"]


if __name__ == "__main__":
    pass
