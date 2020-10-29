import os
import sys
import json
import pickle
import pdb
import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm
from fastai.tabular import *

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

    def train(self, xy_train, features, target, cat_features=[], pct_valid=0.15, n_trees=3000, esr=25):
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


class FastaiModel(object):
    def __init__(self, cat_features, num_features, target):
        self.cat_features = cat_features
        self.num_features = num_features
        self.target = target
        self.model = None

    def prepare_data(self, xy_train, xy_test, valid_pct=0.15):
        procs = [FillMissing, Categorify, Normalize]
        test_data = (
            TabularList.from_df(xy_test, cat_names=self.cat_features, cont_names=self.num_features, procs=procs))

        fast_data = (TabularList
                     .from_df(xy_train, cat_names=self.cat_features, cont_names=self.num_features, procs=procs)
                     .split_by_rand_pct(valid_pct=valid_pct, seed=42)
                     .label_from_df(cols=self.target)
                     .add_test(test_data)
                     .databunch())
        return fast_data

    def train(self, xy_train, xy_test):
        fast_data = self.prepare_data(xy_train, xy_test)
        learn = tabular_learner(fast_data, layers=[256, 128], emb_drop=0.2, metrics=mae)
        learn.fit_one_cycle(4, 1e-4)
        self.model = learn

        # get training results
        tr = learn.validate(learn.data.train_dl)
        va = learn.validate(learn.data.valid_dl)
        print("The Metrics used In Evaluating The Network: {}".format(learn.metrics))
        print("The Training Set Loss: {}".format(tr))
        print("The Validation Set Loss: {}".format(va))

        # get test set predictions
        # test_predictions = learn.get_preds(ds_type=DatasetType.Test)[0]
        # xy_test["fastai_pred"] = test_predictions
        return xy_test

    def predict(self, xy_scoring):
        n_ex = len(xy_scoring)
        fast_scores = []
        for idx in tqdm(range(n_ex)):
            _, _, this_pred = self.model.predict(xy_scoring.iloc[idx])
            fast_scores.append(this_pred.item())
        # xy_scoring["fastai_pred"] = fast_scores
        return fast_scores


def train_lgbm_reg_model(XY_train, XY_test, XY_scoring, features_dict, target="reg_target"):
    features = features_dict["features"]
    cat_features = features_dict["cat_features"]

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
    model.train(XY_train, features, target, cat_features=cat_features)
    df_imp = model.get_feature_importance()

    print(df_imp.head(30))

    return model


def train_fastai_model(XY_train, XY_test, XY_scoring, features_dict, target="reg_target"):
    features = features_dict["features"]
    cat_features = features_dict["cat_features"]
    num_features = features_dict["num_features"]

    model = FastaiModel(cat_features, num_features, target)
    model.train(XY_train, XY_test)
    return model


def train_potential_model():
    pass


def generate_leads():
    XY_train, XY_test, XY_scoring, features_dict = load_data()
    features = features_dict["features"]
    cat_features = features_dict["cat_features"]
    num_features = features_dict["num_features"]

    lgbm_model = train_lgbm_reg_model(XY_train, XY_test, XY_scoring, features_dict, "reg_target")
    fastai_model = train_fastai_model(XY_train, XY_test, XY_scoring, features_dict, "reg_target")

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
    player_id_cost_map = data_maker.get_player_id_cost_map()
    player_id_play_chance_map = data_maker.get_player_id_play_chance_map()
    player_id_selection_map = data_maker.get_player_id_selection_map()
    player_id_ave_points_map = data_maker.get_player_id_ave_points_map()

    df_leads = pd.DataFrame()
    df_leads["player_id"] = XY_scoring["player_id"].values
    df_leads["name"] = df_leads["player_id"].apply(lambda x: player_id_player_name_map.get(x, x))
    df_leads["team"] = df_leads["player_id"].apply(lambda x: team_id_team_name_map[player_id_team_id_map.get(x, x)])
    df_leads["next_opponent"] = XY_scoring["opp_team_id"].apply(lambda x: team_id_team_name_map.get(x, x))
    df_leads["position"] = df_leads["player_id"].apply(lambda x: player_id_player_position_map.get(x, x))
    df_leads["chance_of_play"] = df_leads["player_id"].apply(lambda x: player_id_play_chance_map.get(x, x))
    df_leads["cost"] = df_leads["player_id"].apply(lambda x: player_id_cost_map.get(x, x))
    df_leads["selection_pct"] = df_leads["player_id"].apply(lambda x: player_id_selection_map.get(x, x))
    df_leads["ave_pts"] = df_leads["player_id"].apply(lambda x: player_id_ave_points_map.get(x, x))

    # Predictions
    # pdb.set_trace()
    lgbm_preds = lgbm_model.predict(XY_scoring, features)
    df_leads['lgbm_pred'] = lgbm_preds

    fastai_preds = fastai_model.predict(XY_scoring)
    df_leads['fastai_pred'] = fastai_preds

    return df_leads


if __name__ == "__main__":
    df_fpl = generate_leads()
    df_fpl.to_csv("./data/model_data/predictions.csv", index=False)
