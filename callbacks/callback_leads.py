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
    from scripts.data_loader import DataLoader
    from scripts.data_processor import DataProcessor
    from scripts.data_scrape import DataScraper
    from scripts.utils import load_config
    from app import cache
    from scripts.model_data_ingestion import DataIngestor
except:
    raise ImportError

TIMEOUT = 3600


@cache.memoize(timeout=TIMEOUT)
def ingest_data():
    config_2020 = {"season": "2020_21",
                   "source_dir": "./data",
                   "ingest_dir": "./data/model_data/2020_21/",
                   "player_ingest_filename": "players_raw.csv",
                   "team_ingest_filename": "teams.csv",
                   "gw_ingest_filename": "merged_gw.csv",
                   "understat_ingest_filename": "understat_team_data.pkl",
                   "fixture_ingest_filename": "fixtures.csv"
                   }
    data_ingestor = DataIngestor(config_2020)
    data_ingestor.ingest_player_data()
    data_ingestor.ingest_team_data()
    data_ingestor.ingest_fixture_data()
    data_ingestor.ingest_understat_data()
    data_ingestor.ingest_gw_data()
    result = html.P("Done!")
    return result


# Ingest data callbacks
@app.callback(Output('data-ingest-point', 'children'),
              [Input('ingest-btn-point', 'n_clicks')],
              prevent_initial_call=True)
def execute_data_ingestion_point(n_clicks):
    print("Ingest click xPoint={}".format(n_clicks))
    if n_clicks:
        return ingest_data()
    else:
        return None


@app.callback(Output('data-ingest-potential', 'children'),
              [Input('ingest-btn-potential', 'n_clicks')],
              prevent_initial_call=True)
def execute_data_ingestion_potential(n_clicks):
    print("Ingest click xPotential={}".format(n_clicks))
    if n_clicks:
        return ingest_data()
    else:
        return None


@app.callback(Output('data-ingest-return', 'children'),
              [Input('ingest-btn-return', 'n_clicks')],
              prevent_initial_call=True)
def execute_data_ingestion_return(n_clicks):
    print("Ingest click={}".format(n_clicks))
    if n_clicks:
        return ingest_data()
    else:
        return None
