import pdb
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from app import app

from layouts.layout_utils import make_table, make_dropdown, make_line_plot
from scripts.data_loader import DataLoader
from scripts.data_processor import DataProcessor
from scripts.data_scrape import DataScraper
from scripts.utils import load_config

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


@app.callback([Output('league-standing-table', 'children'),
               Output('league-standing-memory', 'data')],
              [Input('league-search-dropdown', 'value')])
def make_league_standing_table(league_id):
    if not league_id:
        return "", None

    config = load_config()
    data_processor = DataProcessor(config)
    data_loader = DataLoader(config)
    # pdb.set_trace()
    data_processor.save_classic_league_standing(league_id)
    df = data_loader.get_league_standings(league_id)

    try:
        df = df.rename(columns={"entry_name": "Team",
                                "manager_name": "Manager",
                                "rank": "Rank",
                                "score": "Points",
                                "gw_points": "GW Score"})
        df["Rank Delta"] = (df["previous_rank"] - df["Rank"]).astype(int)
        df = df[["Team", "Manager", "Points", "GW Score", "Rank Delta"]].copy()
        df = df.sort_values(by="Points", ascending=False).iloc[:100]  # keep top 100
        table = make_table(df)
        return table, df.to_dict('records')
    except:
        return html.Div("Warning! This League Info Not Found!"), None


@app.callback(Output('team-selection-div', 'children'),
              [Input('league-search-dropdown', 'value'),
               Input('league-standing-memory', 'data')])
def make_team_selection_section(league_id, league_data):
    if league_data:
        df_managers = pd.DataFrame(league_data)
        managers = df_managers["Team"].unique().tolist()
        manager_options = [{'label': this_manager, 'value': this_manager} for this_manager in managers]
        dropdown_id = 'team-selection-dropdown'
        dropdown_section = make_dropdown(dropdown_id, manager_options,
                                         placeholder="Select Teams For Comparison ...", multi_flag=True)
        return dropdown_section

    else:
        return None


@app.callback(Output('league-point-history', 'children'),
              [Input('team-selection-dropdown', 'value')],
              [State('league-search-dropdown', 'value')])
def make_gw_history_plot(teams, league_id):

    def format_hover(x):
        team_name, ovr, val = x
        text = "{}\nOVR: {:.1f}M\nBudget: {:.1f}".format(team_name, ovr/1e6, val/10)
        return text

    if not league_id:
        return ""

    config = load_config()
    data_scraper = DataScraper(config)
    data_processor = DataProcessor(config)
    data_loader = DataLoader(config)

    # data_processor.save_classic_league_history(league_id)
    df = data_loader.get_league_gw_history(league_id)
    start_gw = data_scraper.get_league_start_gameweek(league_id)
    current_gw = data_scraper.get_next_gameweek_id()-1

    data = []
    for team in teams:
        tmp_df = df[df["entry_name"] == team].copy()
        tmp_df = tmp_df[tmp_df["event"] >= start_gw].copy()
        tmp_df = tmp_df.sort_values(by="event")
        tmp_df["league_points"] = tmp_df["points"].cumsum()
        tmp_df["hover_text"] = tmp_df[["entry_name", "overall_rank", "value"]].apply(lambda x: format_hover(x), axis=1)
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
                           hovertemplate='%{hovertext}<br>Score: %{y}' + '<br>%{text}</br>',
                           text=['OVR: {:.1f}M \n Budget: {:.1f}'.format(ovr/1e6, val/10) for ovr, val in zip(tmp_df["overall_rank"].values, tmp_df["value"].values)]
                           )
        # trace = go.Scatter(tmp_df, x="event", y="league_points")
        data.append(trace)
    fig = make_line_plot(data, xlabel='Gameweek', ylabel='Total Score')
    x_start = max(start_gw, current_gw-5)
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
