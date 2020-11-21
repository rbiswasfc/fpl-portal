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
    from scripts.utils import load_config, check_cache_validity
    from app import cache
    from scripts.data_preparation import ModelDataMaker
    from scripts.model_data_ingestion import DataIngestor
    from scripts.feature_engineering import make_XY_data
    from scripts.models import load_data, train_lgbm_model, train_fastai_model
    from callbacks.callback_cache import query_league_data, query_league_history_data, query_league_start_gw, \
        CONFIG_2020, get_league_eo, get_top_eo, load_leads_current_gw, query_manager_current_gw_picks, load_leads
except:
    raise ImportError


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
        title="Expected Scores Predictions",
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
