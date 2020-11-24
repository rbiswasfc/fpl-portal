import os
import pdb
import pulp
import random
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

try:
    from layouts.layout_utils import make_table, make_dropdown, make_line_plot
    from scripts.data_loader import DataLoader
    from scripts.data_processor import DataProcessor
    from scripts.data_scrape import DataScraper
    from scripts.utils import load_config, check_cache_validity
    from app import cache
    from scripts.data_preparation import ModelDataMaker
    from scripts.model_data_ingestion import DataIngestor
    from scripts.feature_engineering import make_XY_data
    from scripts.models import load_data, train_lgbm_model, train_fastai_model
    from callbacks.callback_cache import load_leads, CONFIG_2020, load_all_point_predictions
except:
    raise ImportError


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
    player_team = dict(zip(df["name"], df["team"]))
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
                # 'team': player_team[p],
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


def transfer_optimizer(df_leads, manager_id, num_transfers, model_name):
    df_leads["name"] = df_leads["name"].apply(lambda x: str(x).encode('ascii', 'ignore'))
    config = load_config()
    data_loader = DataLoader(config)
    df_team = pd.DataFrame(data_loader.get_manager_current_gw_picks(manager_id))
    df_team = df_team.rename(columns={"element": "player_id"})
    bank = data_loader.get_manager_bank_balance(manager_id)

    df_cost = df_leads[["player_id", "cost", "name", model_name]].copy()
    df_team = pd.merge(df_team, df_cost, how='inner', on='player_id')
    prev_score = df_team[model_name].sum()

    budget = df_team["cost"].sum() + bank

    # print(df_team.head())
    # print(df_leads.head())
    # print(budget)

    # optimization

    df = df_leads.copy()
    df = df.pipe(add_position_dummy)
    df = df.pipe(add_team_dummy)
    players = df["name"].unique().tolist()
    current_players = df_team["name"].unique().tolist()
    fpl_problem = pulp.LpProblem('FPL_Transfers', pulp.LpMaximize)

    x = pulp.LpVariable.dict('x_ % s', players, lowBound=0, upBound=1, cat=pulp.LpInteger)
    # player score data
    player_points = dict(zip(df["name"], np.array(df[model_name])))
    # objective function
    fpl_problem += sum([player_points[i] * x[i] for i in players])
    # constraints
    position_names = ['gk', 'def', 'mid', 'fwd']
    formation = '2-5-5-3'
    position_constraints = [int(i) for i in formation.split('-')]
    constraints = dict(zip(position_names, position_constraints))
    constraints['total_cost'] = budget
    constraints['team'] = 3
    constraints["num_keep"] = 15 - num_transfers

    # could get straight from dataframe...
    player_cost = dict(zip(df["name"], df["cost"]))
    player_position = dict(zip(df["name"], df["position"]))
    player_team = dict(zip(df["name"], df["team"]))
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
    fpl_problem += sum([x[i] for i in current_players]) == constraints['num_keep']

    # team constraints
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
                # 'team': player_team[p],
                'position': player_position[p],
                'cost': player_cost[p],
                'points': player_points[p]
            })

    solution_info = {
        'formation': formation,
        'total_points': total_points,
        'total_cost': total_cost
    }
    # pdb.set_trace()
    df_squad = pd.DataFrame(optimal_squad)
    now_score = df_squad["points"].sum()
    new_squad = set(df_squad["name"].unique().tolist())
    current_players = set(current_players)
    transfer_in = list(new_squad.difference(current_players))
    transfer_out = list(current_players.difference(new_squad))
    transfer_in = [in_player.decode('utf-8') for in_player in transfer_in]
    transfer_out = [out_player.decode('utf-8') for out_player in transfer_out]
    df_res = pd.DataFrame()
    gain = [0 for i in range(len(transfer_in))]
    gain[-1] = now_score - prev_score
    df_res["Transfer In"] = transfer_in
    df_res["Transfer Out"] = transfer_out
    df_res["gain"] = gain
    df_res["gain"] = df_res["gain"].round(2)
    df_res["gain"] = df_res["gain"].astype(str)
    df_res["gain"] = df_res["gain"].apply(lambda y: "" if int(float(y)) == 0 else y)
    df_res = df_res.rename(columns={"gain": "Gain"})
    return df_res


@app.callback([Output('player-compare-output', 'children'),
               Output('player-prediction-compare-output', 'children')],
              [Input('player-selection-dropdown-a', 'value'),
               Input('player-selection-dropdown-b', 'value'),
               Input('gw-selection-dropdown-squad', 'value')],
              prevent_initial_call=True)
def execute_player_comparison(player_a, player_b, gw_id):
    if not player_a:
        msg = html.P("Please select first player")
        return msg, ""
    if not player_b:
        msg = html.P("Please select second player")
        return msg, ""
    if not gw_id:
        msg = html.P("Please select gameweek in left layout")
        return msg, ""
    #
    df_leads = load_leads(gw_id)
    df_preds = load_all_point_predictions(gw_id)

    # normalization
    pot_div = 12
    point_div = 6
    retrun_div = 0.8

    df_leads["LGBM Potential"] = df_leads["LGBM Potential"] / pot_div
    df_leads["Fast Potential"] = df_leads["Fast Potential"] / pot_div
    df_leads["LGBM Point"] = df_leads["LGBM Point"] / point_div
    df_leads["Fast Point"] = df_leads["Fast Point"] / point_div
    df_leads["LGBM Return"] = df_leads["LGBM Return"] / 0.8
    df_leads["Fast Return"] = df_leads["Fast Return"] / 0.4
    df_leads["Net"] = df_leads["Net"] / 0.4
    df_leads["Cost"] = df_leads["cost"] / 10.0

    df_a = df_leads[df_leads["name"] == player_a].copy()
    df_b = df_leads[df_leads["name"] == player_b].copy()
    df_a_xpts = df_preds[df_preds["name"] == player_a].copy()
    df_b_xpts = df_preds[df_preds["name"] == player_b].copy()

    df_a_xpts = df_a_xpts.sort_values(by="gw")
    df_b_xpts = df_b_xpts.sort_values(by="gw")
    df_a_xpts["xpts"] = df_a_xpts["xpts"].round(2)
    df_b_xpts["xpts"] = df_b_xpts["xpts"].round(2)

    df_a_xpts["size"] = df_a_xpts["pts"] * 3
    df_a_xpts["size"] = df_a_xpts["size"].fillna(2)
    df_a_xpts["size"] = df_a_xpts["size"].astype(float)
    df_a_xpts["size"] = df_a_xpts["size"] + 5

    df_b_xpts["size"] = df_b_xpts["pts"] * 3
    df_b_xpts["size"] = df_b_xpts["size"].fillna(2)
    df_b_xpts["size"] = df_b_xpts["size"].astype(float)
    df_b_xpts["size"] = df_b_xpts["size"] + 5

    keep_cols = ["LGBM Point", "LGBM Potential", "LGBM Return",
                 "Fast Point", "Fast Potential", "Fast Return", "Cost"]
    df_a = df_a[keep_cols].copy().T.reset_index()
    df_a.columns = ["theta", "r"]

    df_b = df_b[keep_cols].copy().T.reset_index()
    df_b.columns = ["theta", "r"]

    # pdb.set_trace()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=df_a['r'].values, theta=df_a["theta"].values,
                                  fill='toself', name=player_a))
    fig.add_trace(go.Scatterpolar(r=df_b['r'].values, theta=df_b["theta"].values,
                                  fill='toself', name=player_b))
    fig.update_layout(polar=dict(radialaxis=dict(visible=False)), showlegend=True)
    # fig = px.line_polar(df_a, r='r', theta='theta', line_close=True)
    radar_graph = dcc.Graph(figure=fig)

    # Compare player predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(name=player_a,
                             x=df_a_xpts["gw"].values,
                             y=df_a_xpts["xpts"].values,
                             marker_size=df_a_xpts['size'],
                             hovertext=df_a_xpts['name'],
                             hoverlabel=dict(namelength=0),
                             hovertemplate='%{hovertext}<br>Pred: %{y} | GW: %{x}' + '<br>%{text}</br>',
                             text=['Opponent: {} | Pts: {}| Flag: {}'.format(a, b, c) for a, b, c in
                                   zip(df_a_xpts["opponent"].values, df_a_xpts["pts"].values,
                                       df_a_xpts["chance_of_play"].values)]
                             ))
    fig.add_trace(go.Scatter(name=player_b,
                             x=df_b_xpts["gw"].values,
                             y=df_b_xpts["xpts"].values,
                             marker_size=df_b_xpts['size'],
                             hovertext=df_b_xpts['name'],
                             hoverlabel=dict(namelength=0),
                             hovertemplate='%{hovertext}<br>Pred: %{y}' + '<br>%{text}</br>',
                             text=['Opponent: {} | Pts: {}| Flag: {}'.format(a, b, c) for a, b, c in
                                   zip(df_b_xpts["opponent"].values, df_b_xpts["pts"].values,
                                       df_b_xpts["chance_of_play"].values)]
                             ))
    fig.layout.template = 'seaborn'
    layout = go.Layout(xaxis={'title': 'Gameweek'},
                       yaxis={'title': 'xPts'},
                       margin={'l': 5, 'b': 75, 't': 25, 'r': 5},
                       hovermode='x')
    fig.update_layout(layout)
    fig.update_layout(
        title="LGBM Next Gameweeks Prediction Comparison",
        legend=dict(
            x=0.8,
            y=0.05,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color="black"
            ),
        )
    )

    x_max, x_min = df_a_xpts["gw"].max(), df_a_xpts["gw"].min()
    y_max = max(df_a_xpts["xpts"].max(), df_b_xpts["xpts"].max())
    fig.update_xaxes(range=(x_min, x_max), ticks="inside", tick0=x_min, dtick=1)
    fig.update_yaxes(range=(0, y_max + 0.25), ticks="inside", tick0=0, dtick=1)

    fig.add_shape(type="line", x0=gw_id, y0=0, x1=gw_id, y1=y_max + 0.25,
                  line=dict(color="LightSeaGreen", width=4, dash="dash"))

    fig.add_trace(go.Scatter(x=[gw_id - 0.5], y=[y_max / 2.0],
                             text=["Next GW"], mode="text", showlegend=False))
    pred_graph = dcc.Graph(figure=fig)

    return radar_graph, pred_graph


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
def execute_squad_optimization(n_clicks, gw_id, model_name, formation, squad_val, bench_val, uncertain_flag):
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
        col_map = {"name": "Player", "team": "Team", "cost": "Cost", "position": "Position", "points": model_name}
        df_squad_xi = df_squad_xi.rename(columns=col_map)
        df_squad_bench = df_squad_bench.rename(columns=col_map)
        position_map = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}

        df_squad_xi["pos_map"] = df_squad_xi["Position"].apply(lambda x: position_map[x])
        df_squad_bench["pos_map"] = df_squad_bench["Position"].apply(lambda x: position_map[x])
        df_squad_xi = df_squad_xi.sort_values(by=["pos_map"])
        df_squad_bench = df_squad_bench.sort_values(by=["pos_map"])
        df_squad_xi = df_squad_xi.drop(columns=["pos_map"])
        df_squad_bench = df_squad_bench.drop(columns=["pos_map"])
        table_xi, table_bench = make_table(df_squad_xi, page_size=11), make_table(df_squad_bench)

        return table_xi, table_bench
    else:
        return html.P("Button Not Clicked!")


@app.callback(Output('transfer-suggestion-output', 'children'),
              [Input('transfer-optimization-btn', 'n_clicks')],
              [State('manager-selection-transfers', 'value'),
               State('transfer-selection-numbers', 'value'),
               State('gw-selection-dropdown-squad', 'value'),
               State('model-selection-dropdown-optim', 'value')],
              prevent_initial_call=True)
def execute_transfer_suggestions(n_clicks, manager_id, num_transfers, gw_id, model_name):
    if not manager_id:
        msg = html.P("Please select manager...")
        return msg
    if not num_transfers:
        msg = html.P("Please select number of transfer to be made...")
        return msg
    if not gw_id:
        msg = html.P("Please select GW for scoring...")
        return msg
    if not model_name:
        msg = html.P("Please select ML Model...")
        return msg

    if n_clicks:
        tables = []
        n_suggestions = 1
        df_leads = load_leads(gw_id)

        for i in range(n_suggestions):
            try:
                df_transfer = transfer_optimizer(df_leads, manager_id, num_transfers, model_name)
                tables.append(make_table(df_transfer))
                exclude_names = df_transfer["Transfer In"].unique().tolist()
                df_leads = df_leads[~df_leads["name"].isin(exclude_names)].copy()
            except:
                pass

        output = html.Div(
            children=tables
        )

        return output

    return html.P("Button Not Clicked!")


@app.callback([Output('transfer-analyzer-output', 'children'),
               Output('captaincy-analyzer-output', 'children')],
              [Input('manager-selection-transfer-analyzer', 'value')],
              prevent_initial_call=True)
def execute_squad_analyzer(manager_id):
    if not manager_id:
        msg = html.P("Please select manager...")
        return msg

    forward_window = 5
    config = load_config()
    data_scraper = DataScraper(config)
    data_loader = DataLoader(config)
    data_maker = ModelDataMaker(CONFIG_2020)
    player_id_player_name_map = data_maker.get_player_id_player_name_map()

    next_gw = data_scraper.get_next_gameweek_id()
    picks_data = data_scraper.get_entry_gw_picks_history(manager_id, next_gw - 1)
    dfs = []
    gw_num_transfer_map = dict()
    gw_penalty_map = dict()
    for this_pick in picks_data:
        gw_id = this_pick["entry_history"]["event"]
        num_transfer = this_pick["entry_history"]["event_transfers"]
        penalty = this_pick["entry_history"]["event_transfers_cost"]
        gw_num_transfer_map[gw_id] = num_transfer
        gw_penalty_map[gw_id] = penalty
        picks = this_pick["picks"]
        tmp_df = pd.DataFrame(picks)
        tmp_df["gw"] = gw_id
        dfs.append(tmp_df)

    df_picks = pd.concat(dfs)
    df_fpl = pd.read_csv("./data/model_data/2020_21/merged_gw.csv")
    df_fpl = df_fpl[["element", "gw", "total_points"]].copy()
    df_picks = pd.merge(df_picks, df_fpl, how='left', on=["element", "gw"])
    df_picks["total_points"] = df_picks["total_points"].fillna(0)

    data = data_scraper.get_entry_gw_transfers(manager_id)
    df_transfer = pd.DataFrame(data)
    gameweeks = df_transfer["event"].unique().tolist()
    summary_dfs = []
    for this_gw in gameweeks:
        tmp_df = df_transfer[df_transfer["event"] == this_gw].copy()

        tmp_df = tmp_df.drop_duplicates(subset=["element_in", "element_out"])
        in_out_map = dict()
        for idx, row in tmp_df.iterrows():
            transfer_in, transfer_out = row["element_in"], row["element_out"]
            in_out_map[transfer_in] = transfer_out
        transfer_in = tmp_df["element_in"].unique().tolist()
        cond_a = (df_picks["gw"] >= this_gw) & (df_picks["gw"] < this_gw + forward_window)
        cond_b = df_picks["element"].isin(transfer_in)
        df_focus = df_picks[cond_a & cond_b].copy()
        df_focus = df_focus.rename(columns={"element": "element_in", "total_points": "in_points"})
        df_focus["element_out"] = df_focus["element_in"].apply(lambda x: in_out_map[x])
        df_focus = pd.merge(df_focus, df_fpl, how='left', left_on=["element_out", "gw"], right_on=["element", "gw"])
        df_focus = df_focus.drop(columns=["element"])
        df_focus = df_focus.rename(columns={"total_points": "out_points"})
        df_focus["out_points"] = df_focus["out_points"].fillna(0)
        df_focus["impact"] = (df_focus["in_points"] - df_focus["out_points"]) * df_focus["multiplier"]
        df_summary = df_focus.groupby("element_in")["impact"].agg('sum').reset_index()
        df_summary["gw"] = this_gw
        df_summary["element_out"] = df_summary["element_in"].apply(lambda x: in_out_map[x])
        summary_dfs.append(df_summary)
    df_final = pd.concat(summary_dfs)
    df_final["Transfer In"] = df_final["element_in"].apply(lambda x: player_id_player_name_map.get(x, x))
    df_final["Transfer Out"] = df_final["element_out"].apply(lambda x: player_id_player_name_map.get(x, x))
    df_final["GW"] = df_final["gw"].astype(int)
    df_final["Delta"] = df_final["impact"].astype(int)
    df_final = df_final[["GW", "Transfer In", "Transfer Out", "Delta"]].copy()
    table = make_table(df_final)

    df_final = df_final.groupby("GW")["Delta"].agg('sum').reset_index()
    df_final["# Transfers"] = df_final["GW"].apply(lambda x: gw_num_transfer_map[x])
    df_final["Hits"] = df_final["GW"].apply(lambda x: gw_penalty_map[x])
    df_final = df_final[["GW", "# Transfers", "Hits", "Delta"]].copy()
    table_meta = make_table(df_final)

    transfer_output_section = html.Div(children=[
        table,
        html.Div("", style={"margin-top": "2rem"}),
        table_meta
    ])

    def get_score(element, gw):
        this_df = df_fpl[(df_fpl["element"] == element) & (df_fpl["gw"] == gw)].copy()
        if len(this_df) == 0:
            return 'NA'
        else:
            score = this_df["total_points"].values[0]
            return int(score)

    # captaincy output
    df_meta = data_loader.get_gameweek_metadata()
    df_meta = df_meta[["id", "top_element", "most_captained"]].copy()
    # df_meta["top_element_points"] = df_meta["top_element_info"].apply(lambda x: x["points"] if x else 'NA')
    # df_meta = df_meta.drop(columns=["top_element_info"])
    df_meta = df_meta.rename(columns={"id": "gw"})
    df_captain = df_picks[df_picks["multiplier"] == 2].copy()
    df_captain = pd.merge(df_captain, df_meta, how='left', on='gw')
    df_captain["Cap"] = df_captain["element"].apply(lambda x: player_id_player_name_map.get(x, x))
    df_captain["Cap Score"] = df_captain["total_points"]
    df_captain["GW"] = df_captain["gw"]
    df_captain["Popular Cap"] = df_captain["most_captained"].apply(lambda x: player_id_player_name_map.get(int(x), x))
    df_captain["Popular Cap Score"] = df_captain[["most_captained", "gw"]].apply(lambda x: get_score(int(x[0]), int(x[1])), axis=1)
    df_captain["Top Player"] = df_captain["top_element"].apply(lambda x: player_id_player_name_map.get(x, x))
    df_captain["Top Score"] = df_captain[["top_element", "gw"]].apply(lambda x: get_score(int(x[0]), int(x[1])), axis=1)
    df_captain = df_captain[["GW", "Cap", "Popular Cap", "Top Player", "Cap Score", "Popular Cap Score", "Top Score"]].copy()
    cap_table = make_table(df_captain)

    return transfer_output_section, cap_table
