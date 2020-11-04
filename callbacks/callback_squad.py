import os
import pdb
import pulp
import shap
from pathlib import Path
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from app import app
from tqdm import tqdm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from fastai.tabular import load_learner

try:
    from layouts.layout_utils import make_table, make_dropdown, make_line_plot
    from scripts.data_loader import DataLoader, check_cache_validity
    from scripts.data_processor import DataProcessor
    from scripts.data_scrape import DataScraper
    from scripts.utils import load_config
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

TIMEOUT = 3600


def add_position_dummy(df):
    for p in df.position.unique():
        df['is_' + str(p).lower()] = np.where(df.position == p, int(1), int(0))
    return df


def add_team_dummy(df):
    for t in df.team.unique():
        df['team_' + str(t).lower()] = np.where(df.team == t, int(1), int(0))
    return df


def squad_optimizer(df, formation, budget=100.0, optimise_on='LGBM Point'):
    df = df.pipe(add_position_dummy)
    df = df.pipe(add_team_dummy)
    players = df["name"].unique().tolist()
    fpl_problem = pulp.LpProblem('FPL', pulp.LpMaximize)
    # create a dictionary of pulp variables with keys from names
    x = pulp.LpVariable.dict('x_ % s', players, lowBound=0, upBound=1, cat=pulp.LpInteger)
    # player score data
    player_points = dict(zip(df["name"], np.array(df[optimise_on])))
    # objective function
    fpl_problem += sum([player_points[i] * x[i] for i in players])
    # constraints
    position_names = ['gk', 'def', 'mid', 'fwd']
    position_constraints = [int(i) for i in formation.split('-')]
    constraints = dict(zip(position_names, position_constraints))
    constraints['total_cost'] = budget
    constraints['team'] = 3
    # could get straight from dataframe...
    player_cost = dict(zip(df["name"], df["cost"]))
    player_position = dict(zip(df["name"], df["position"]))
    player_gk = dict(zip(df["name"], df["is_gk"]))
    player_def = dict(zip(df["name"], df["is_def"]))
    player_mid = dict(zip(df["name"], df["is_mid"]))
    player_fwd = dict(zip(df["name"], df["is_fwd"]))
    # apply the constraints
    fpl_problem += sum([player_cost[i] * x[i] for i in players]) <= float(constraints['total_cost'])
    fpl_problem += sum([player_gk[i] * x[i] for i in players]) == constraints['gk']
    fpl_problem += sum([player_def[i] * x[i] for i in players]) == constraints['def']
    fpl_problem += sum([player_mid[i] * x[i] for i in players]) == constraints['mid']
    fpl_problem += sum([player_fwd[i] * x[i] for i in players]) == constraints['fwd']
    for t in df.team:
        player_team = dict(zip(df["name"], df['team_' + str(t).lower()]))
        fpl_problem += sum([player_team[i] * x[i] for i in players]) <= constraints['team']
    # solve the thing
    fpl_problem.solve()

    total_points = 0.
    total_cost = 0.
    optimal_squad = []

    for p in players:
        if x[p].value() != 0:
            total_points += player_points[p]
            total_cost += player_cost[p]

            optimal_squad.append({
                'name': p,
                'position': player_position[p],
                'cost': player_cost[p],
                'points': player_points[p]
            })

    solution_info = {
        'formation': formation,
        'total_points': total_points,
        'total_cost': total_cost
    }
    df_squad = pd.DataFrame(optimal_squad)
    df_squad = df_squad.sort_values(by=['position', 'points'], ascending=False)
    return df_squad, solution_info


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
        if not check_cache_validity(file_path, valid_days=2.0):
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
    return df_leads


@app.callback([Output('squad-optim-output-play-xi', 'children'),
               Output('squad-optim-output-bench', 'children')],
              [Input('squad-optimization-btn', 'n_clicks')],
              [State('gw-selection-dropdown-squad', 'value'),
               State('model-selection-dropdown-optim', 'value'),
               State('formation-selection-dropdown-squad', 'value'),
               State('squad-value-input', 'value'),
               State('bench-value-input', 'value'),
               State('uncertain-flag', 'value')],
              prevent_initial_call=True)
def execute_fastai_return_scoring(n_clicks, gw_id, model_name, formation, squad_val, bench_val, uncertain_flag):
    if not gw_id:
        msg = html.P("Please select GW for scoring")
        return msg, msg

    if not model_name:
        msg = html.P("Please select Model")
        return msg, msg

    if not formation:
        msg = html.P("Please select Formation")
        return msg, msg

    if not squad_val:
        msg = html.P("Please select Squad Value")
        return msg, msg

    if not bench_val:
        msg = html.P("Please select Bench Value")
        return msg, msg

    if not uncertain_flag:
        msg = html.P("Please select Uncertain Flag")
        return msg, msg
    df_leads = load_leads(gw_id)
    # pdb.set_trace()
    df_leads["name"] = df_leads["name"].apply(lambda x: str(x).encode('ascii', 'ignore'))
    print(df_leads.head())
    if n_clicks:
        df_squad_xi, sol_info_xi = squad_optimizer(df_leads, formation=formation,
                                                   budget=squad_val - bench_val, optimise_on=model_name)
        xi_players = [int(i) for i in formation.split('-')]
        bench_players = [str(2 - xi_players[0]), str(5 - xi_players[1]), str(5 - xi_players[2]), str(3 - xi_players[3])]
        bench_formation = "-".join(bench_players)
        xi_names = df_squad_xi["name"].unique().tolist()
        df_leads = df_leads[~df_leads["name"].isin(xi_names)].copy()
        df_squad_bench, sol_info_bench = squad_optimizer(df_leads, formation=bench_formation,
                                                         budget=bench_val, optimise_on=model_name)

        df_squad_xi["name"] = df_squad_xi["name"].apply(lambda x: x.decode('utf-8'))
        df_squad_bench["name"] = df_squad_bench["name"].apply(lambda x: x.decode('utf-8'))
        # df_squad = df_squad[["position", "cost", "points"]].copy()
        df_squad_xi["points"] = df_squad_xi["points"].round(2)
        df_squad_bench["points"] = df_squad_bench["points"].round(2)
        col_map = {"name": "Player", "cost": "Cost", "position": "Position", "points": model_name}
        df_squad_xi = df_squad_xi.rename(columns=col_map)
        df_squad_bench = df_squad_bench.rename(columns=col_map)
        table_xi, table_bench = make_table(df_squad_xi, page_size=11), make_table(df_squad_bench)

        return table_xi, table_bench
    else:
        return html.P("Button Not Clicked!")
