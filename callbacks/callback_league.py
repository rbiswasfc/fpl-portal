import pdb
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from app import app

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

try:
    from layouts.layout_utils import make_table, make_dropdown, make_line_plot
    from layouts.layout_leads import query_next_gameweek
    from scripts.data_loader import DataLoader
    from scripts.data_processor import DataProcessor
    from scripts.data_scrape import DataScraper
    from scripts.utils import load_config
    from app import cache
    from scripts.data_preparation import ModelDataMaker
    from scripts.model_data_ingestion import DataIngestor
    from scripts.feature_engineering import make_XY_data
    from scripts.models import load_data, train_lgbm_model, train_fastai_model
    from callbacks.callback_squad import load_leads
except:
    raise ImportError

TIMEOUT = 3600 * 2

CONFIG_2020 = {
    "data_dir": "./data/model_data/2020_21/",
    "file_fixture": "fixtures.csv",
    "file_team": "teams.csv",
    "file_gw": "merged_gw.csv",
    "file_player": "players_raw.csv",
    "file_understat_team": "understat_team_data.pkl",
    "scoring_gw": "NA"
}


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


@app.callback([Output('league-standing-table', 'children'),
               Output('league-standing-memory', 'data')],
              [Input('league-search-dropdown', 'value')])
def make_league_standing_table(league_id):
    if not league_id:
        return "", None
    df = query_league_data(league_id)

    try:
        df = df.rename(columns={"entry_name": "Team",
                                "manager_name": "Manager",
                                "rank": "Rank",
                                "score": "Points",
                                "gw_points": "GW Score"})
        df["Rank Delta"] = (df["previous_rank"] - df["Rank"]).astype(int)
        df = df.sort_values(by="Points", ascending=False).iloc[:100]  # keep top 100
        df_clean = df[["Team", "Manager", "Points", "GW Score", "Rank Delta"]].copy()
        table = make_table(df_clean)
        return table, df.to_dict('records')
    except:
        return html.Div("Warning! This League Info Not Found!"), None


@app.callback([Output('team-selection-dropdown', 'options'),
               Output('league-team-picks-dropdown', 'options')],
              [Input('league-search-dropdown', 'value'),
               Input('league-standing-memory', 'data')])
def make_team_selection_section(league_id, league_data):
    if league_id and league_data:
        df_managers = pd.DataFrame(league_data)
        managers = df_managers["Team"].unique().tolist()
        manager_options = [{'label': this_manager, 'value': this_manager} for this_manager in managers]

        return manager_options, manager_options

    else:
        return [], []


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


@app.callback([Output('league-team-picks-display', 'children'),
               Output('league-team-xp-output', 'children')],
              [Input('league-standing-memory', 'data'),
               Input('league-team-picks-dropdown', 'value'),
               State('league-search-dropdown', 'value')])
def show_current_team_picks(league_data, team_name, league_id):
    if not league_data:
        return "", ""
    if not team_name:
        return "", ""
    if not league_id:
        return "", ""

    if league_data and team_name:
        df_managers = pd.DataFrame(league_data)
        df_tmp = df_managers[df_managers["Team"] == team_name].copy()
        manager_id = df_tmp["entry_id"].unique().tolist()[0]
        df = query_manager_current_gw_picks(manager_id, league_id)
        team_points = (df["Points"] * df["multiplier"]).sum()
        expected_points = (df["xP"] * df["multiplier"]).sum()

        summary_section = html.Div(
            children=[
                html.P("Expected Points: {:.2f}".format(expected_points), className='col-6'),
                html.P("Live Points: {}".format(team_points), className='col-6'),
            ],
            className='row',
        )
        keep_cols = ["Player", "Team", "Position", "Top EO", "League EO", "xP", "Points"]
        df = df[keep_cols].copy()
        table = make_table(df, page_size=11)
        return table, summary_section


@app.callback(Output('league-point-history', 'children'),
              [Input('team-selection-dropdown', 'value')],
              [State('league-search-dropdown', 'value')])
def make_gw_history_plot(teams, league_id):
    if (not league_id) or (not teams):
        return ""

    df = query_league_history_data(league_id)
    start_gw = query_league_start_gw(league_id)
    current_gw = query_next_gameweek() - 1

    data = []
    for team in teams:
        tmp_df = df[df["entry_name"] == team].copy()
        tmp_df = tmp_df[tmp_df["event"] >= start_gw].copy()
        tmp_df = tmp_df.sort_values(by="event")
        tmp_df["league_points"] = tmp_df["points"].cumsum()
        tmp_df["gain"] = (tmp_df["overall_rank"] - tmp_df["overall_rank"].shift(periods=1)).fillna(0)

        # x_values = tmp_df["event"].tolist()
        # x_values.insert(0, x_values[0]-1)
        # y_values = tmp_df["league_points"].tolist()
        # y_values.insert(0, 0)
        print(tmp_df)
        trace = go.Scatter(x=tmp_df["event"],
                           y=tmp_df["league_points"],
                           name=team,
                           marker_size=tmp_df['points'],
                           hovertext=tmp_df['entry_name'],
                           hoverlabel=dict(namelength=0),
                           hovertemplate='%{hovertext}<br>Score: %{y} | GW Score: %{marker.size:,}' + '<br>%{text}</br>',
                           text=['OVR: {:.1f}M | Budget: {:.1f} | Gain: {:.1f}k'.format(ovr / 1e6, val / 10,
                                                                                        -delta / 1000) for
                                 ovr, val, delta in zip(tmp_df["overall_rank"].values,
                                                        tmp_df["value"].values, tmp_df["gain"].values)]
                           )
        # trace = go.Scatter(tmp_df, x="event", y="league_points")
        data.append(trace)
    fig = make_line_plot(data, xlabel='Gameweek', ylabel='Total Score')
    x_start = max(start_gw, current_gw - 5)
    fig.update_xaxes(range=(x_start, current_gw + 0.05), ticks="inside", tick0=x_start, dtick=1)
    fig.update_yaxes(ticks="inside")
    fig.update_layout(
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

    graph = dcc.Graph(figure=fig)
    return graph
