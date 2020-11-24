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


def load_data(gw, dataset_dir="./data/model_data/xy_data/"):
    """
    Load data for modelling
    :param gw: scoring gw
    :type gw: int
    :param dataset_dir: folder with XY data
    :type dataset_dir: str
    :return: train, test, scoring dataframe and features
    :rtype: tuple
    """
    # TODO: exception handing when reading XY data
    XY_train = pd.read_csv(os.path.join(dataset_dir, "xy_train_gw_{}.csv".format(gw)))
    XY_test = pd.read_csv(os.path.join(dataset_dir, "xy_test_gw_{}.csv".format(gw)))
    XY_scoring = pd.read_csv(os.path.join(dataset_dir, "xy_scoring_gw_{}.csv".format(gw)))

    with open(os.path.join(dataset_dir, "features_after_fe.pkl"), 'rb') as f:
        features_dict = pickle.load(f)
    return XY_train, XY_test, XY_scoring, features_dict


def remove_next_features(feat_list):
    """
    Remove next features for model
    :param feat_list: list of features used in model
    :type feat_list: List
    :return: List of features without next features
    :rtype: List
    """
    cur_feats = [feat for feat in feat_list if 'next' not in feat]
    return cur_feats


class LgbModel(object):
    """
    Constructor for Light GBM models
    """

    def __init__(self, params):
        """
        initialization for model class
        :param params: lgb model params
        :type params: dict
        """
        self.params = params
        self.model = None
        self.features = None
        self.target = None
        self.model_output_dir = "./data/model_outputs/"
        check_create_dir(self.model_output_dir)

    def train(self, xy_train, features, target, cat_features=None, pct_valid=0.15, n_trees=3000, esr=25, seed=42):
        """
        Train lgb model
        :param seed:
        :type seed:
        :param xy_train: training data
        :type xy_train: pd.DataFrame
        :param features: features list
        :type features: List
        :param target: target column
        :type target: str
        :param cat_features: categorical features
        :type cat_features: List
        :param pct_valid: size of validation set
        :type pct_valid: float
        :param n_trees: number of trees in lgb model
        :type n_trees: int
        :param esr: early stopping rounds
        :type esr: int
        :return: evaluation results
        :rtype: dict
        """
        if cat_features is None:
            cat_features = []
        self.target = target
        X, y = xy_train[features].copy(), xy_train[target].values
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=pct_valid, random_state=seed)
        train_data = lgbm.Dataset(X_train, label=y_train, feature_name=features)
        valid_data = lgbm.Dataset(X_valid, label=y_valid, feature_name=features, reference=train_data)
        evaluation_results = {}
        self.model = lgbm.train(self.params,
                                train_data,
                                num_boost_round=n_trees,
                                valid_sets=[train_data, valid_data],
                                evals_result=evaluation_results,
                                early_stopping_rounds=esr,
                                categorical_feature=cat_features,
                                verbose_eval=100
                                )
        self.features = features
        print("Training Done ...")
        return evaluation_results

    def predict(self, df):
        """
        Get model predictions
        :param df: scoring dataframe
        :type df: pd.DataFrame
        :return: scores
        :rtype: np.ndarray
        """
        preds = self.model.predict(df[self.features], num_iteration=self.model.best_iteration)
        return preds

    def get_feature_importance(self):
        """
        Get model feature importance
        :return: feature importance dataframe
        :rtype: pd.DataFrame
        """
        df_imp = pd.DataFrame({'imp': self.model.feature_importance(importance_type='gain'),
                               'feature_name': self.model.feature_name()})
        df_imp = df_imp.sort_values(by='imp', ascending=False).reset_index(drop=True)
        return df_imp

    def save_model(self):
        """
        Save lgb model
        """
        model = self.model
        save_path = os.path.join(self.model_output_dir, "lgbm_{}_model.txt".format(self.target))
        model.save_model(save_path, num_iteration=model.best_iteration)

    def load_model(self):
        """
        Load lgb model
        :return: model
        :rtype: lgb.Booster
        """
        save_path = os.path.join(self.model_output_dir, "lgbm_{}_model.txt".format(self.target))
        model = lgbm.Booster(model_file=save_path)
        return model


class FastaiModel(object):
    """
    FastAI model constructor
    """

    def __init__(self, cat_features, num_features, target):
        """
        Fast AI model constructor
        :param cat_features: list of categorical features
        :type cat_features: List
        :param num_features: list of numerical features
        :type num_features: List
        :param target: target columns
        :type target: str
        """
        self.cat_features = cat_features
        self.num_features = num_features
        self.target = target
        self.model = None

        self.model_output_dir = "./data/model_outputs/"
        check_create_dir(self.model_output_dir)

    def prepare_data(self, xy_train, xy_test, valid_pct=0.15):
        """
        Prepare data loader for fastai model
        :param xy_train: training data
        :type xy_train: pd.DataFrame
        :param xy_test: test data
        :type xy_test: pd.DataFrame
        :param valid_pct: validation data size
        :type valid_pct: float
        :return: data bunch for fastai model
        :rtype: data bunch
        """
        procs = [FillMissing, Categorify, Normalize]
        test_data = (
            TabularList.from_df(xy_test, cat_names=self.cat_features, cont_names=self.num_features, procs=procs))

        fast_data = (TabularList
                     .from_df(xy_train, cat_names=self.cat_features, cont_names=self.num_features, procs=procs)
                     .split_by_rand_pct(valid_pct=valid_pct, seed=42)
                     .label_from_df(cols=self.target)
                     .databunch(bs=32))
        return fast_data

    def train(self, xy_train, xy_test):
        """
        Train fast ai model
        :param xy_train: training data
        :type xy_train: pd.DataFrame
        :param xy_test: test data
        :type xy_test: pd.DataFrame
        :return: training loss, validation loss
        :rtype: tuple
        """
        if self.target == "star_target":
            metric_fn = accuracy
        else:
            metric_fn = mae
        fast_data = self.prepare_data(xy_train, xy_test)
        learn = tabular_learner(fast_data, layers=[256, 128], emb_drop=0.2, metrics=metric_fn)
        learn.fit_one_cycle(4, 1e-4)  # set this to 4
        self.model = learn

        train_loss, valid_loss = learn.recorder.losses, learn.recorder.val_losses
        # save model
        save_dir = os.path.join(self.model_output_dir, "fastai_{}_model".format(self.target))
        check_create_dir(save_dir)
        save_path = os.path.join(save_dir, "export.pkl")
        learn.export(file=save_path)
        return train_loss, valid_loss

    def predict(self, xy_scoring):
        """
        fast ai model predictions
        :param xy_scoring: scoring dataframe
        :type xy_scoring: pd.DataFrame
        :return: scores
        :rtype: np.ndarray
        """
        n_ex = len(xy_scoring)
        fast_scores = []
        for idx in tqdm(range(n_ex)):
            _, _, this_pred = self.model.predict(xy_scoring.iloc[idx])
            fast_scores.append(this_pred.item())
        return fast_scores


def train_lgbm_model(gw, target="reg_target", params=None):
    """
    Train light gbm model
    :param params:
    :type params:
    :param gw: scoring gameweek
    :type gw: int
    :param target: target column
    :type target: str
    :return: trained model, training results
    :rtype: tuple
    """
    try:
        XY_train, XY_test, XY_scoring, features_dict = load_data(gw)
    except:
        print("ERROR: No data found for modelling")
        return None

    features = features_dict["features"]
    cat_features = features_dict["cat_features"]

    if target != "pot_target":
        print("# features before removing next features = {}".format(len(features)))
        print("# cat features before removing next features = {}".format(len(cat_features)))
        features = remove_next_features(features)
        cat_features = remove_next_features(cat_features)
        print("# features after removing next features = {}".format(len(features)))
        print("# cat features after removing next features = {}".format(len(cat_features)))
    if not params:
        if target != "star_target":
            params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': 'l1',
                'learning_rate': 0.01,
                'feature_fraction': 0.75,
                'bagging_fraction': 0.75,
                'verbose': -1,
                "max_depth": 7,
                "num_leaves": 15,
                "max_bin": 64
            }
        else:
            params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'learning_rate': 0.01,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'verbose': -1,
                "max_depth": 7,
                "num_leaves": 15,
                "max_bin": 64,
                'is_unbalance': 'true'
            }

    print("Shape before removing null targets: {}".format(XY_train.shape))
    XY_train = XY_train[~XY_train[target].isna()].copy()
    print("Shape after removing null targets: {}".format(XY_train.shape))

    if target == "reg_target":
        # make soft target
        amp = 0.5
        XY_train[target] = XY_train[target].apply(lambda y: y + (np.random.rand() - 0.5) * amp)
    seed = np.random.randint(1, 1000)

    model = LgbModel(params)
    evaluation_results = model.train(XY_train, features, target, cat_features=cat_features, seed=seed)
    model.save_model()
    df_imp = model.get_feature_importance()
    df_imp.to_csv(os.path.join(model.model_output_dir, "lgbm_{}_feature_imp.csv".format(target)), index=False)

    print(df_imp.head(30))

    return model, evaluation_results


def train_fastai_model(gw, target="reg_target"):
    """
    Train fastai model
    :param gw: scoring gameweek
    :type gw: int
    :param target: target column
    :type target: str
    :return: loss history
    :rtype: List
    """
    try:
        XY_train, XY_test, XY_scoring, features_dict = load_data(gw)
    except:
        print("ERROR: No data found for modelling")
        return None

    features = features_dict["features"]
    cat_features = features_dict["cat_features"]
    num_features = features_dict["num_features"]

    if target != "pot_target":
        print("# features before removing next features = {}".format(len(features)))
        print("# cat features before removing next features = {}".format(len(cat_features)))
        print("# num features before removing next features = {}".format(len(num_features)))
        features = remove_next_features(features)
        cat_features = remove_next_features(cat_features)
        num_features = remove_next_features(num_features)
        print("# features after removing next features = {}".format(len(features)))
        print("# cat features after removing next features = {}".format(len(cat_features)))
        print("# num features after removing next features = {}".format(len(num_features)))

    print("Shape before removing null targets: {}".format(XY_train.shape))
    XY_train = XY_train[~XY_train[target].isna()].copy()
    print("Shape after removing null targets: {}".format(XY_train.shape))

    model = FastaiModel(cat_features, num_features, target)
    train_loss, valid_loss = model.train(XY_train, XY_test)
    loss_history = dict()
    loss_history["train"] = train_loss
    loss_history["valid"] = valid_loss
    return loss_history


if __name__ == "__main__":
    pass
    # df_fpl = generate_leads(7)
    # df_fpl.to_csv("./data/model_data/predictions.csv", index=False)
