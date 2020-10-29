import os
import sys
import json
import pickle
import pdb
import numpy as np
import pandas as pd
import pdb

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import lightgbm as lgbm

sys.path.insert(0, './')

try:
    from scripts.utils import load_config, check_create_dir
    from scripts.data_preparation import ModelDataMaker
    from scripts.data_scrape import DataScraper
    from scripts.feature_engineering import FeatureEngineering
except:
    raise ImportError


def load_data(dataset_dir="./data/model_data/xy_data/"):
    XY_train = pd.read_csv(os.path.join(dataset_dir, "xy_train.csv"))
    XY_test = pd.read_csv(os.path.join(dataset_dir, "xy_test.csv"))
    XY_scoring = pd.read_csv(os.path.join(dataset_dir, "xy_scoring.csv"))

    with open(os.path.join(dataset_dir, "features_after_fe.pkl"), 'rb') as f:
        features_dict = pickle.load(f)
    return XY_train, XY_test, XY_scoring, features_dict


class LgbModel(object):
    def __init__(self, params):
        self.params = params
        self.model = None

    def train(self, xy_train, features, target, cat_features=[], pct_valid=0.2, n_trees=3000, esr=25):
        X, y = xy_train[features].copy(), xy_train[target].values
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=pct_valid, random_state=42)
        train_data = lgbm.Dataset(X_train, label=y_train, feature_name=features)
        valid_data = lgbm.Dataset(X_valid, label=y_valid, feature_name=features, reference=train_data)
        self.model = lgbm.train(self.params,
                                train_data,
                                num_boost_round=n_trees,
                                valid_sets=[valid_data],
                                early_stopping_rounds=esr,
                                categorical_feature=cat_features,
                                verbose_eval=100
                                )
        print("Training Done ...")

    def predict(self, df, features):
        preds = self.model.predict(df[features], num_iteration=self.model.best_iteration)
        return preds

    def get_feature_importance(self):
        df_imp = pd.DataFrame({'imp': self.model.feature_importance(importance_type='gain'),
                               'feature_name': self.model.feature_name()})
        df_imp = df_imp.sort_values(by='imp', ascending=False).reset_index(drop=True)
        return df_imp


if __name__ == "__main__":
    XY_train, XY_test, XY_scoring, features_dict = load_data()
    features = features_dict["features"]
    cat_features = features_dict["cat_features"]
    print(cat_features)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'l1',
        'learning_rate': 0.005,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'verbose': -1,
        "max_depth": 7,
        "num_leaves": 31,
        "max_bin": 64
    }

    model = LgbModel(params)
    model.train(XY_train, features, "reg_target", cat_features=cat_features)
    df_imp = model.get_feature_importance()
    print(df_imp.head(40))



    # scoring

    config_2020 = {
        "data_dir": "./data/model_data/2020_21/",
        "file_fixture": "fixtures.csv",
        "file_team": "teams.csv",
        "file_gw": "merged_gw.csv",
        "file_player": "players_raw.csv",
        "file_understat_team": "understat_team_data.pkl",
        "scoring_gw": "NA"
    }

    data_maker = ModelDataMaker(config_2020)
    player_id_team_id_map = data_maker.get_player_id_team_id_map()
    player_id_player_name_map = data_maker.get_player_id_player_name_map()
    player_id_player_position_map = data_maker.get_player_id_player_position_map()
    team_id_team_name_map = data_maker.get_team_id_team_name_map()

    preds = model.predict(XY_test, features)
    df_res = pd.DataFrame()
    df_res["player_id"] = XY_test["player_id"].values
    df_res["name"] = XY_test["player_id"].apply(lambda x: player_id_player_name_map.get(x, x))
    df_res["team"] = XY_test["player_id"].apply(lambda x: team_id_team_name_map[player_id_team_id_map.get(x, x)])
    df_res["position"] = XY_test["player_id"].apply(lambda x: player_id_player_position_map.get(x, x))
    df_res['y_true'] = XY_test["total_points"].values
    df_res['y_pred'] = preds
    df_res = df_res.sort_values(by='y_pred', ascending=False)
    print(df_res.head())

    # scoring
    preds = model.predict(XY_scoring, features)
    df_leads = pd.DataFrame()
    df_leads["player_id"] = XY_scoring["player_id"].values
    df_leads["name"] = df_leads["player_id"].apply(lambda x: player_id_player_name_map.get(x, x))
    df_leads["team"] = df_leads["player_id"].apply(lambda x: team_id_team_name_map[player_id_team_id_map.get(x, x)])
    df_leads["opponent"] = XY_scoring["opp_team_id"].apply(lambda x: team_id_team_name_map.get(x, x))
    df_leads["position"] = df_leads["player_id"].apply(lambda x: player_id_player_position_map.get(x, x))
    df_leads['y_true'] = XY_scoring["total_points"].values
    df_leads['y_pred'] = preds
    df_leads = df_leads.sort_values(by='y_pred', ascending=False)
    print(df_leads.head())

    pdb.set_trace()

