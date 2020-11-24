import os
import numpy as np
import pandas as pd
import dash_html_components as html
from tqdm import tqdm
from fastai.tabular import load_learner
from pathlib import Path
import shap
from itertools import product
import pdb

try:
    from scripts.data_loader import DataLoader
    from scripts.data_processor import DataProcessor
    from scripts.data_scrape import DataScraper
    from scripts.utils import load_config, check_cache_validity
    from app import cache
    from scripts.data_preparation import ModelDataMaker
    from scripts.model_data_ingestion import DataIngestor
    from scripts.feature_engineering import make_XY_data
    from scripts.models import load_data, train_lgbm_model, train_fastai_model
except:
    raise ImportError

CONFIG_2020 = {
    "data_dir": "./data/model_data/2020_21/",
    "file_fixture": "fixtures.csv",
    "file_team": "teams.csv",
    "file_gw": "merged_gw.csv",
    "file_player": "players_raw.csv",
    "file_understat_team": "understat_team_data.pkl",
    "scoring_gw": "NA"
}

TIMEOUT = 3600 * 48
TIMEOUT_MID = 3600 * 6
TIMEOUT_SHORT = 3600 * 0.5


@cache.memoize(timeout=TIMEOUT_MID)
def ingest_data():
    ingest_config = {"season": "2020_21",
                     "source_dir": "./data",
                     "ingest_dir": "./data/model_data/2020_21/",
                     "player_ingest_filename": "players_raw.csv",
                     "team_ingest_filename": "teams.csv",
                     "gw_ingest_filename": "merged_gw.csv",
                     "understat_ingest_filename": "understat_team_data.pkl",
                     "fixture_ingest_filename": "fixtures.csv"
                     }
    data_ingestor = DataIngestor(ingest_config)
    data_ingestor.ingest_player_data()
    data_ingestor.ingest_team_data()
    data_ingestor.ingest_fixture_data()
    data_ingestor.ingest_understat_data()
    data_ingestor.ingest_gw_data()
    result = html.P("Done!", style={"text-align": "center"})
    return result


@cache.memoize(timeout=TIMEOUT)
def prepare_xy_model_data(gw):
    n_next_gws = 7
    for this_gw in range(gw, gw + n_next_gws):
        make_XY_data(this_gw)
    result = html.P("Done!", style={"text-align": "center"})
    return result


# LGBM Model Training
@cache.memoize(timeout=TIMEOUT)
def perform_lgbm_point_training(gw, params=None):
    model, evaluation_results = train_lgbm_model(gw, target="reg_target", params=params)
    return model, evaluation_results


@cache.memoize(timeout=TIMEOUT)
def perform_lgbm_potential_training(gw):
    model, evaluation_results = train_lgbm_model(gw, target="pot_target")
    return model, evaluation_results


@cache.memoize(timeout=TIMEOUT)
def perform_lgbm_return_training(gw):
    model, evaluation_results = train_lgbm_model(gw, target="star_target")
    return model, evaluation_results


# Fast AI Model Training
@cache.memoize(timeout=TIMEOUT)
def perform_fastai_point_training(gw):
    loss_history = train_fastai_model(gw, target="reg_target")
    return loss_history


@cache.memoize(timeout=TIMEOUT)
def perform_fastai_potential_training(gw):
    loss_history = train_fastai_model(gw, target="pot_target")
    return loss_history


@cache.memoize(timeout=TIMEOUT)
def perform_fastai_return_training(gw):
    loss_history = train_fastai_model(gw, target="star_target")
    return loss_history


# scoring
@cache.memoize(timeout=TIMEOUT)
def perform_lgbm_point_scoring(gw):
    XY_train, XY_test, XY_scoring, features_dict = load_data(gw)
    # ensemble approach
    task = ['train']
    boosting_type = ['gbdt']
    objective = ['regression']
    metric = ['l1']
    learning_rate = [0.02, 0.01, 0.005]
    feature_fraction = [0.75]
    bagging_fraction = [0.75]
    verbose = [-1]
    max_depth = [6, 7, 8]
    num_leaves = [15]
    max_bin = [64]
    iter_params = product(task, boosting_type, objective, metric,
                          learning_rate, feature_fraction, bagging_fraction,
                          verbose, max_depth, num_leaves, max_bin)

    res_dict = dict()
    for i, param_val in enumerate(iter_params):
        print("==" * 50)
        print("Model instance: {}".format(i))
        params = {
            'task': param_val[0],
            'boosting_type': param_val[1],
            'objective': param_val[2],
            'metric': param_val[3],
            'learning_rate': param_val[4],
            'feature_fraction': param_val[5],
            'bagging_fraction': param_val[6],
            'verbose': param_val[7],
            "max_depth": param_val[8],
            "num_leaves": param_val[9],
            "max_bin": param_val[10]
        }
        print(params)
        model, _ = perform_lgbm_point_training(gw, params)
        preds = model.predict(XY_scoring)
        df = pd.DataFrame()
        df["player_id"] = XY_scoring["player_id"].values
        df["gw"] = gw
        df['pred'] = preds
        print("==" * 50)
        tmp_dict = {'params': params, 'result': df}
        res_dict[i] = tmp_dict

    # aggregate
    cnt = 1
    dfs = []
    for k, v in res_dict.items():
        df_res = v['result']
        df_tmp = df_res[["player_id", "pred"]].copy()
        df_tmp = df_tmp.rename(columns={"pred": "pred_{}".format(cnt)})
        cnt = cnt + 1
        dfs.append(df_tmp)

    df_final = dfs[0].copy()
    for i in range(1, len(dfs)):
        df_final = pd.merge(df_final, dfs[i], how='left', on='player_id')

    pred_cols = [col for col in df_final.columns if 'pred_' in col]
    df_final["lgbm_point_pred"] = df_final[pred_cols].apply(lambda x: np.mean(x), axis=1)
    df_final["gw"] = gw
    print(df_final.head())
    df_final = df_final[["player_id", "gw", "lgbm_point_pred"]].copy()

    save_path = os.path.join(model.model_output_dir, "lgbm_point_predictions_gw_{}.csv".format(gw))
    df_final.to_csv(save_path, index=False)

    result = html.P("Done!", style={"text-align": "center"})
    return result


@cache.memoize(timeout=TIMEOUT)
def perform_lgbm_potential_scoring(gw):
    model, _ = perform_lgbm_potential_training(gw)
    XY_train, XY_test, XY_scoring, features_dict = load_data(gw)
    preds = model.predict(XY_scoring)
    df = pd.DataFrame()
    df["player_id"] = XY_scoring["player_id"].values
    df["gw"] = gw
    df['lgbm_potential_pred'] = preds
    save_path = os.path.join(model.model_output_dir, "lgbm_potential_predictions_gw_{}.csv".format(gw))
    df.to_csv(save_path, index=False)
    print(df.head())
    result = html.P("Done!", style={"text-align": "center"})
    return result


@cache.memoize(timeout=TIMEOUT)
def perform_lgbm_return_scoring(gw):
    model, _ = perform_lgbm_return_training(gw)
    XY_train, XY_test, XY_scoring, features_dict = load_data(gw)
    preds = model.predict(XY_scoring)
    df = pd.DataFrame()
    df["player_id"] = XY_scoring["player_id"].values
    df["gw"] = gw
    df['lgbm_return_pred'] = preds
    save_path = os.path.join(model.model_output_dir, "lgbm_return_predictions_gw_{}.csv".format(gw))
    df.to_csv(save_path, index=False)
    print(df.head())
    result = html.P("Done!", style={"text-align": "center"})
    return result


# fastai point predictor
@cache.memoize(timeout=TIMEOUT)
def perform_fastai_point_scoring(gw):
    export_dir = Path("./data/model_outputs/fastai_reg_target_model")
    learn = load_learner(export_dir)
    XY_train, XY_test, XY_scoring, features_dict = load_data(gw)

    n_ex = len(XY_scoring)
    fast_scores = []
    for idx in tqdm(range(n_ex)):
        _, _, this_pred = learn.predict(XY_scoring.iloc[idx])
        fast_scores.append(this_pred.item())

    df = pd.DataFrame()
    df["player_id"] = XY_scoring["player_id"].values
    df["gw"] = gw
    df['fastai_point_pred'] = fast_scores
    save_path = os.path.join("./data/model_outputs", "fastai_point_predictions_gw_{}.csv".format(gw))
    df.to_csv(save_path, index=False)
    print(df.head())
    result_code = 1
    return result_code


@cache.memoize(timeout=TIMEOUT)
def perform_fastai_potential_scoring(gw):
    export_dir = Path("./data/model_outputs/fastai_pot_target_model")
    learn = load_learner(export_dir)
    XY_train, XY_test, XY_scoring, features_dict = load_data(gw)

    n_ex = len(XY_scoring)
    fast_scores = []
    for idx in tqdm(range(n_ex)):
        _, _, this_pred = learn.predict(XY_scoring.iloc[idx])
        fast_scores.append(this_pred.item())

    df = pd.DataFrame()
    df["player_id"] = XY_scoring["player_id"].values
    df["gw"] = gw
    df['fastai_potential_pred'] = fast_scores
    save_path = os.path.join("./data/model_outputs", "fastai_potential_predictions_gw_{}.csv".format(gw))
    df.to_csv(save_path, index=False)
    print(df.head())
    result_code = 1
    return result_code


@cache.memoize(timeout=TIMEOUT)
def perform_fastai_return_scoring(gw):
    export_dir = Path("./data/model_outputs/fastai_star_target_model")
    learn = load_learner(export_dir)
    XY_train, XY_test, XY_scoring, features_dict = load_data(gw)

    n_ex = len(XY_scoring)
    fast_scores = []
    for idx in tqdm(range(n_ex)):
        _, _, this_pred = learn.predict(XY_scoring.iloc[idx])
        fast_scores.append(this_pred[1].item())

    df = pd.DataFrame()
    df["player_id"] = XY_scoring["player_id"].values
    df["gw"] = gw
    df['fastai_return_pred'] = fast_scores
    save_path = os.path.join("./data/model_outputs", "fastai_return_predictions_gw_{}.csv".format(gw))
    df.to_csv(save_path, index=False)
    print(df.head())
    result_code = 1
    return result_code


# SHAP
@cache.memoize(timeout=TIMEOUT)
def perform_shap_analysis(model_name, gw_id):
    XY_train, XY_test, XY_scoring, features_dict = load_data(gw_id)
    model, features = None, None
    if model_name == 'LGBM Point':
        model, _ = perform_lgbm_point_training(gw_id)
        features = model.features
    if model_name == 'LGBM Potential':
        model, _ = perform_lgbm_potential_training(gw_id)
        features = model.features
    if model_name == 'LGBM Return':
        model, _ = perform_lgbm_return_training(gw_id)
        features = model.features

    X_scoring = XY_scoring[features].copy()
    explainer = shap.TreeExplainer(model.model)

    shap_values = explainer.shap_values(X_scoring)
    ave_score = explainer.expected_value
    if model_name == 'LGBM Return':
        shap_values = shap_values[1]
        ave_score = explainer.expected_value[1]
    df = pd.DataFrame(shap_values)
    shap_cols = ["shap_" + feat for feat in features]
    df.columns = shap_cols
    df_exp = pd.concat([XY_scoring, df], axis=1)

    data_maker = ModelDataMaker(CONFIG_2020)
    player_id_player_name_map = data_maker.get_player_id_player_name_map()
    df_exp["name"] = df_exp["player_id"].apply(lambda x: player_id_player_name_map.get(x, x))

    return df_exp, ave_score


@cache.memoize(timeout=TIMEOUT)
def query_league_data(league_id):
    config = load_config()
    data_loader = DataLoader(config)
    df = data_loader.get_league_standings(league_id)
    return df


@cache.memoize(timeout=TIMEOUT)
def query_league_history_data(league_id):
    config = load_config()
    data_scraper = DataScraper(config)
    data_processor = DataProcessor(config)
    data_loader = DataLoader(config)

    data_processor.save_classic_league_history(league_id)
    df = data_loader.get_league_gw_history(league_id)
    return df


@cache.memoize(timeout=TIMEOUT)
def query_league_start_gw(league_id):
    config = load_config()
    data_scraper = DataScraper(config)
    start_gw = data_scraper.get_league_start_gameweek(league_id)
    return start_gw


@cache.memoize(timeout=TIMEOUT)
def get_league_eo(league_id):
    config = load_config()
    data_loader = DataLoader(config)
    print(league_id)
    df_league = data_loader.get_league_standings(league_id)
    managers = df_league["entry_id"].unique().tolist()
    dfs = []
    for manager in managers:
        df = pd.DataFrame(data_loader.get_manager_current_gw_picks(manager))
        dfs.append(df)
    df_eo = pd.concat(dfs)
    n_players = int(len(df_eo) / 15.0)
    df_stats = df_eo.groupby('element')["multiplier"].agg("sum").reset_index()
    df_stats["League EO"] = df_stats["multiplier"] * 100.0 / n_players
    df_stats["League EO"] = df_stats["League EO"].round(2)
    df_stats = df_stats[["element", "League EO"]].copy()
    return df_stats


@cache.memoize(timeout=TIMEOUT)
def get_top_eo():
    config = load_config()
    data_loader = DataLoader(config)
    df_top = data_loader.get_top_manager_picks()
    n_players = int(len(df_top) / 15.0)
    df_stats = df_top.groupby('element')["multiplier"].agg("sum").reset_index()
    df_stats["Top EO"] = df_stats["multiplier"] * 100.0 / n_players
    df_stats["Top EO"] = df_stats["Top EO"].round(2)
    df_stats = df_stats[["element", "Top EO"]].copy()
    return df_stats


@cache.memoize(timeout=TIMEOUT)
def load_leads_current_gw():
    config = load_config()
    data_loader = DataLoader(config)
    current_gw = data_loader.get_next_gameweek_id() - 1
    print(current_gw)
    df_leads = load_leads(current_gw)
    print(df_leads)
    try:
        print(df_leads.head())
    except:
        # no scores available
        df_leads = pd.DataFrame()
        df_leads["player_id"] = [-1, -1]
        df_leads["LGBM Point"] = [-1, -1]
        df_leads["Fast Point"] = [-1, -1]

    df_leads['xP'] = (df_leads["LGBM Point"] + df_leads["Fast Point"]) / 2.0
    df_leads['xP'] = df_leads['xP'].round(2)
    return df_leads


@cache.memoize(timeout=TIMEOUT)
def query_manager_current_gw_picks(manager_id, league_id):
    config = load_config()
    data_loader = DataLoader(config)
    data = data_loader.get_manager_current_gw_picks(manager_id)
    df = pd.DataFrame(data)

    data_maker = ModelDataMaker(CONFIG_2020)
    player_id_team_id_map = data_maker.get_player_id_team_id_map()
    player_id_player_name_map = data_maker.get_player_id_player_name_map()
    player_id_player_position_map = data_maker.get_player_id_player_position_map()
    team_id_team_name_map = data_maker.get_team_id_team_name_map()
    player_id_cost_map = data_maker.get_player_id_cost_map()
    player_id_selection_map = data_maker.get_player_id_selection_map()

    # points
    df_gw = data_loader.get_live_gameweek_data()
    df_gw = df_gw.rename(columns={"id": "element", "event_points": "Points"})
    # print(df_gw.head(1).T)
    df_gw = df_gw[["element", "Points"]].copy()
    df_gw = df_gw.drop_duplicates(subset=["element"])
    df = pd.merge(df, df_gw, how='left', on="element")
    # print(df.head())
    df["Player"] = df["element"].apply(lambda x: player_id_player_name_map.get(x, x))
    df["Player"] = df["Player"].apply(lambda x: " ".join(x.split(" ")[:2]))
    df["Team"] = df["element"].apply(lambda x: team_id_team_name_map[player_id_team_id_map[x]])
    df["Position"] = df["element"].apply(lambda x: player_id_player_position_map.get(x, x))
    df["Player"] = df[["Player", "is_captain"]].apply(lambda x: x[0] + " (C)" if x[1] else x[0], axis=1)
    df["Player"] = df[["Player", "is_vice_captain"]].apply(lambda x: x[0] + " (VC)" if x[1] else x[0], axis=1)
    df["Cost"] = df["element"].apply(lambda x: player_id_cost_map.get(x, x))
    df["Cost"] = df["Cost"] / 10
    df["TSB"] = df["element"].apply(lambda x: player_id_selection_map.get(x, x))

    # Get Effective ownership
    df_stats = get_top_eo()
    df_league_eo = get_league_eo(league_id)
    df = pd.merge(df, df_stats, on="element", how="left")
    df = pd.merge(df, df_league_eo, on="element", how="left")

    df_leads = load_leads_current_gw()
    df_leads = df_leads[["player_id", "xP"]].copy()
    df = pd.merge(df, df_leads, how='left', left_on="element", right_on="player_id")

    position_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    df["pos"] = df["Position"].apply(lambda x: position_map[x])
    df = df.sort_values(by=["pos"])
    df_xi = df[df["multiplier"] > 0].copy()
    df_bench = df[df["multiplier"] == 0].copy()
    df = pd.concat([df_xi, df_bench])
    # print(df.head())
    keep_cols = ["Player", "multiplier", "Team", "Position", "Top EO", "League EO", "xP", "Points"]
    # keep_cols = ["Player", "Team", "Position", "TSB", "Top EO", "Points"]
    # merge player info
    df = df[keep_cols].copy()
    return df


@cache.memoize(timeout=TIMEOUT)
def load_leads(gw_id):
    data_maker = ModelDataMaker(CONFIG_2020)
    output_dir = "./data/model_outputs/"
    lgbm_point_path = os.path.join(output_dir, "lgbm_point_predictions_gw_{}.csv".format(gw_id))
    lgbm_potential_path = os.path.join(output_dir, "lgbm_potential_predictions_gw_{}.csv".format(gw_id))
    lgbm_return_path = os.path.join(output_dir, "lgbm_return_predictions_gw_{}.csv".format(gw_id))

    fastai_point_path = os.path.join(output_dir, "fastai_point_predictions_gw_{}.csv".format(gw_id))
    fastai_potential_path = os.path.join(output_dir, "fastai_potential_predictions_gw_{}.csv".format(gw_id))
    fastai_return_path = os.path.join(output_dir, "fastai_return_predictions_gw_{}.csv".format(gw_id))
    all_paths = [lgbm_point_path, lgbm_potential_path, lgbm_return_path,
                 fastai_point_path, fastai_potential_path, fastai_return_path]
    dfs = []
    for file_path in all_paths:
        if not check_cache_validity(file_path, valid_days=5.0):
            return html.P("refresh model scores")
        df = pd.read_csv(file_path)
        dfs.append(df)
    XY_train, XY_test, XY_scoring, features_dict = load_data(gw_id)
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
    df_leads["gw"] = gw_id
    df_leads = df_leads.drop_duplicates(subset=["player_id"])

    # merge predictions
    for df in dfs:
        df = df.drop_duplicates()
        df_leads = pd.merge(df_leads, df, how='left', on=['player_id', 'gw'])
    df_leads["cost"] = df_leads["cost"] / 10

    model_name_col_map = {
        "LGBM Point": "lgbm_point_pred",
        "LGBM Potential": "lgbm_potential_pred",
        "LGBM Return": "lgbm_return_pred",
        "Fast Point": "fastai_point_pred",
        "Fast Potential": "fastai_potential_pred",
        "Fast Return": "fastai_return_pred"
    }
    col_model_name_map = dict()
    for k, v in model_name_col_map.items():
        col_model_name_map[v] = k

    df_leads = df_leads.rename(columns=col_model_name_map)
    df_leads["Net"] = (2 * df_leads["LGBM Point"] + df_leads["LGBM Potential"] +
                       2 * df_leads["Fast Point"] + df_leads["Fast Potential"]) * df_leads["Fast Return"] * df_leads[
                          "LGBM Return"]
    max_net = df_leads["Net"].max()
    df_leads["Net"] = df_leads["Net"] / max_net
    return df_leads


# @cache.memoize(timeout=TIMEOUT)
def load_all_point_predictions(gw_id):
    output_dir = "./data/model_outputs/"

    data_maker = ModelDataMaker(CONFIG_2020)
    df_map = data_maker.get_effective_gameweek_map()
    player_id_team_id_map = data_maker.get_player_id_team_id_map()
    player_id_player_name_map = data_maker.get_player_id_player_name_map()
    player_id_player_position_map = data_maker.get_player_id_player_position_map()
    team_id_team_name_map = data_maker.get_team_id_team_name_map()
    player_id_cost_map = data_maker.get_player_id_cost_map()
    player_id_play_chance_map = data_maker.get_player_id_play_chance_map()
    player_id_selection_map = data_maker.get_player_id_selection_map()

    dfs = []
    for this_gw in range(gw_id - 3, gw_id + 7):
        try:
            lgbm_point_path = os.path.join(output_dir, "lgbm_point_predictions_gw_{}.csv".format(this_gw))
            df = pd.read_csv(lgbm_point_path)
            dfs.append(df)
        except:
            print("Scores not found for GW={}".format(this_gw))

    XY_train, _, _, _ = load_data(gw_id)
    XY_train = XY_train[XY_train["season_id"] == 2].copy()
    XY_train = XY_train[["player_id", "gw_id", "total_points"]].copy()
    XY_train = XY_train.rename(columns={"gw_id": "gw"})
    XY_train = XY_train.drop_duplicates(subset=["player_id", "gw"])

    df_preds = pd.concat(dfs)
    df_map = df_map[["gw_id", "own_team_id", "fixture_opp_team_id"]].copy()
    df_map = df_map.rename(columns={"gw_id": "gw", "own_team_id": "team_id"})
    df_map = df_map.drop_duplicates(subset=["gw", "team_id"])
    df_preds["name"] = df_preds["player_id"].apply(lambda x: player_id_player_name_map.get(x, x))
    df_preds["team_id"] = df_preds["player_id"].apply(lambda x: player_id_team_id_map.get(x, x))

    df_preds = pd.merge(df_preds, df_map, how='left', on=["gw", "team_id"])
    df_preds = pd.merge(df_preds, XY_train, how='left', on=["player_id", "gw"])

    df_preds["team"] = df_preds["player_id"].apply(lambda x: team_id_team_name_map[player_id_team_id_map.get(x, x)])
    df_preds["opponent"] = df_preds["fixture_opp_team_id"].apply(lambda x: team_id_team_name_map.get(x, 'NA'))
    df_preds["position"] = df_preds["player_id"].apply(lambda x: player_id_player_position_map.get(x, x))
    df_preds["chance_of_play"] = df_preds["player_id"].apply(lambda x: player_id_play_chance_map.get(x, x))
    df_preds["cost"] = df_preds["player_id"].apply(lambda x: player_id_cost_map.get(x, x)) / 10.0
    df_preds["cost"] = df_preds["cost"].round(1)
    df_preds["selection_pct"] = df_preds["player_id"].apply(lambda x: player_id_selection_map.get(x, x))
    df_preds = df_preds.rename(columns={"lgbm_point_pred": "xpts", "total_points": "pts"})
    df_preds = df_preds.drop_duplicates(subset=["player_id", "gw"])

    keep_cols = ["player_id", "name", "team", "position", "gw", "cost", "chance_of_play", "opponent", "xpts", "pts"]
    df_preds = df_preds[keep_cols].copy()

    return df_preds
