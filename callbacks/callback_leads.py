import os
import pdb
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
    from scripts.data_loader import DataLoader
    from scripts.data_processor import DataProcessor
    from scripts.data_scrape import DataScraper
    from scripts.utils import load_config
    from app import cache
    from scripts.model_data_ingestion import DataIngestor
    from scripts.feature_engineering import make_XY_data
    from scripts.models import load_data, train_lgbm_model, train_fastai_model
except:
    raise ImportError

TIMEOUT = 3600 * 12


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
    result = html.P("Done!", style={"text-align": "center"})
    return result


@cache.memoize(timeout=TIMEOUT)
def prepare_xy_model_data(gw):
    make_XY_data(gw)
    result = html.P("Done!", style={"text-align": "center"})
    return result


# LGBM Model Training
@cache.memoize(timeout=TIMEOUT)
def perform_lgbm_point_training(gw):
    model, evaluation_results = train_lgbm_model(gw, target="reg_target")
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


# Ingest data callbacks
@app.callback(Output('data-ingest-div', 'children'),
              [Input('data-ingest-btn', 'n_clicks')],
              prevent_initial_call=True)
def execute_data_ingestion(n_clicks):
    print("Ingest click={}".format(n_clicks))
    if n_clicks:
        return ingest_data()
    else:
        return None


@app.callback(Output('data-fe-div', 'children'),
              [Input('data-fe-btn', 'n_clicks')],
              [State('gw-selection-dropdown', 'value')],
              prevent_initial_call=True)
def execute_fe(n_clicks, gw):
    print("FE click={}".format(n_clicks))
    if n_clicks:
        if not gw:
            return html.P("Please Select GW")
        print("Preparing data for gameweek={}".format(gw))
        return prepare_xy_model_data(gw)
    else:
        return None


# Model Training Callbacks
@app.callback([Output('lgbm-xnext-outcome', 'children'),
               Output('xnext-feature-imp', 'children')],
              [Input('lgbm-xnext-btn', 'n_clicks')],
              [State('gw-selection-dropdown', 'value')],
              prevent_initial_call=True)
def execute_lgbm_point_training(n_clicks, gw):
    print("LGB Point Model click={}".format(n_clicks))
    if n_clicks:
        if not gw:
            return html.P("Please Select GW"), ""
        print("Training LGBM Point Model for gameweek={}".format(gw))
        model, evaluation_results = perform_lgbm_point_training(gw)
        train_loss, valid_loss = evaluation_results["training"]["l1"], evaluation_results["valid_1"]["l1"]
        iterations = [i for i in range(len(train_loss))]
        # print(iterations)
        train_plot = go.Scatter(x=iterations, y=train_loss, mode='lines', name='Train')
        valid_plot = go.Scatter(x=iterations, y=valid_loss, mode='lines', name='Valid')
        fig = make_line_plot([train_plot, valid_plot], xlabel='# Iterations', ylabel='MAE')
        fig.update_xaxes(range=(0, len(train_loss)), ticks="inside", tick0=0, dtick=500)
        legend = dict(x=0.02, y=0.02, traceorder="normal", font=dict(family="sans-serif", size=12))
        fig.update_layout(legend=legend, title="LGBM Training History")

        graph = dcc.Graph(figure=fig)

        # feature importance
        df_imp = pd.read_csv("./data/model_outputs/lgbm_reg_target_feature_imp.csv")
        print(df_imp.head())
        max_imp = df_imp["imp"].max()
        df_imp["relative_imp"] = df_imp["imp"] / max_imp
        df_imp = df_imp.sort_values(by="imp", ascending=False)
        df_imp["rank"] = [i + 1 for i in range(len(df_imp))]
        top_k = 15
        df_imp = df_imp.iloc[:top_k].copy()
        fig = px.bar(df_imp, y='relative_imp', x='feature_name', title="Point Predictor: Feature Importance",
                     labels={"relative_imp": "Relative Importance", "feature_name": "Feature"},
                     template="seaborn")

        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.update_layout(margin={'l': 5, 'b': 75, 't': 25, 'r': 0})
        imp_bar = dcc.Graph(figure=fig)

        return graph, imp_bar
    else:
        return html.P("Button Not Clicked!", style={"text-align": "center"}), ""


@app.callback([Output('lgbm-xpotential-outcome', 'children'),
               Output('xpotential-feature-imp', 'children')],
              [Input('lgbm-xpotential-btn', 'n_clicks')],
              [State('gw-selection-dropdown', 'value')],
              prevent_initial_call=True)
def execute_lgbm_potential_training(n_clicks, gw):
    print("LGB Potential Model click={}".format(n_clicks))
    if n_clicks:
        if not gw:
            return html.P("Please Select GW"), ""
        print("Training LGBM Potential Model for gameweek={}".format(gw))
        model, evaluation_results = perform_lgbm_potential_training(gw)
        train_loss, valid_loss = evaluation_results["training"]["l1"], evaluation_results["valid_1"]["l1"]
        iterations = [i for i in range(len(train_loss))]
        # print(iterations)
        train_plot = go.Scatter(x=iterations, y=train_loss, mode='lines', name='Train')
        valid_plot = go.Scatter(x=iterations, y=valid_loss, mode='lines', name='Valid')
        fig = make_line_plot([train_plot, valid_plot], xlabel='# Iterations', ylabel='MAE')
        fig.update_xaxes(range=(0, len(train_loss)), ticks="inside", tick0=0, dtick=500)
        legend = dict(x=0.02, y=0.02, traceorder="normal", font=dict(family="sans-serif", size=12))
        fig.update_layout(legend=legend, title="LGBM Training History")

        graph = dcc.Graph(figure=fig)

        # feature importance
        df_imp = pd.read_csv("./data/model_outputs/lgbm_pot_target_feature_imp.csv")
        print(df_imp.head())
        max_imp = df_imp["imp"].max()
        df_imp["relative_imp"] = df_imp["imp"] / max_imp
        df_imp = df_imp.sort_values(by="imp", ascending=False)
        df_imp["rank"] = [i + 1 for i in range(len(df_imp))]
        top_k = 15
        df_imp = df_imp.iloc[:top_k].copy()
        fig = px.bar(df_imp, y='relative_imp', x='feature_name', title="Point Predictor: Feature Importance",
                     labels={"relative_imp": "Relative Importance", "feature_name": "Feature"},
                     template="seaborn")

        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.update_layout(margin={'l': 5, 'b': 75, 't': 25, 'r': 0})
        imp_bar = dcc.Graph(figure=fig)

        return graph, imp_bar
    else:
        return html.P("Button Not Clicked!", style={"text-align": "center"}), ""


@app.callback([Output('lgbm-xreturn-outcome', 'children'),
               Output('xreturn-feature-imp', 'children')],
              [Input('lgbm-xreturn-btn', 'n_clicks')],
              [State('gw-selection-dropdown', 'value')],
              prevent_initial_call=True)
def execute_lgbm_return_training(n_clicks, gw):
    print("LGB Return Model click={}".format(n_clicks))
    if n_clicks:
        if not gw:
            return html.P("Please Select GW"), ""
        print("Training LGBM Return Model for gameweek={}".format(gw))
        model, evaluation_results = perform_lgbm_return_training(gw)
        train_loss, valid_loss = evaluation_results["training"]["l1"], evaluation_results["valid_1"]["l1"]
        iterations = [i for i in range(len(train_loss))]
        # print(iterations)
        train_plot = go.Scatter(x=iterations, y=train_loss, mode='lines', name='Train')
        valid_plot = go.Scatter(x=iterations, y=valid_loss, mode='lines', name='Valid')
        fig = make_line_plot([train_plot, valid_plot], xlabel='# Iterations', ylabel='Log Loss')
        fig.update_xaxes(range=(0, len(train_loss)), ticks="inside", tick0=0, dtick=500)
        legend = dict(x=0.02, y=0.02, traceorder="normal", font=dict(family="sans-serif", size=12))
        fig.update_layout(legend=legend, title="LGBM Training History")

        graph = dcc.Graph(figure=fig)

        # feature importance
        df_imp = pd.read_csv("./data/model_outputs/lgbm_star_target_feature_imp.csv")
        print(df_imp.head())
        max_imp = df_imp["imp"].max()
        df_imp["relative_imp"] = df_imp["imp"] / max_imp
        df_imp = df_imp.sort_values(by="imp", ascending=False)
        df_imp["rank"] = [i + 1 for i in range(len(df_imp))]
        top_k = 15
        df_imp = df_imp.iloc[:top_k].copy()
        fig = px.bar(df_imp, y='relative_imp', x='feature_name', title="Point Predictor: Feature Importance",
                     labels={"relative_imp": "Relative Importance", "feature_name": "Feature"},
                     template="seaborn")

        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.update_layout(margin={'l': 5, 'b': 75, 't': 25, 'r': 0})
        imp_bar = dcc.Graph(figure=fig)

        return graph, imp_bar
    else:
        return html.P("Button Not Clicked!", style={"text-align": "center"}), ""


@app.callback(Output('fastai-xnext-outcome', 'children'),
              [Input('fastai-xnext-btn', 'n_clicks')],
              [State('gw-selection-dropdown', 'value')],
              prevent_initial_call=True)
def execute_fastai_point_training(n_clicks, gw):
    print("FastAI Point Model click={}".format(n_clicks))
    if n_clicks:
        if not gw:
            return html.P("Please Select GW")
        print("Training FastAI Point Model for gameweek={}".format(gw))
        loss_history = perform_fastai_point_training(gw)
        train_loss, valid_loss = loss_history["train"], loss_history["valid"]
        iterations = [i for i in range(len(train_loss))]
        num_valid_steps = len(valid_loss)
        valid_step_size = len(iterations) / num_valid_steps
        valid_iterations = [(i + 1) * valid_step_size for i in range(num_valid_steps)]
        # print(iterations)
        train_plot = go.Scatter(x=iterations, y=train_loss, mode='lines', name='Train')
        valid_plot = go.Scatter(x=valid_iterations, y=valid_loss, mode='lines', name='Valid')
        fig = make_line_plot([train_plot, valid_plot], xlabel='# Batches Processed', ylabel='L2 Loss')
        fig.update_xaxes(range=(0, len(train_loss)), ticks="inside", tick0=0, dtick=1000)
        # fig.update_yaxes(range=(0, 10), ticks="inside", tick0=1)
        legend = dict(x=0.02, y=0.02, traceorder="normal", font=dict(family="sans-serif", size=12))
        fig.update_layout(legend=legend, title="FastAI Training History")

        graph = dcc.Graph(figure=fig)
        return graph
    else:
        return html.P("Button Not Clicked!", style={"text-align": "center"})


@app.callback(Output('fastai-xpotential-outcome', 'children'),
              [Input('fastai-xpotential-btn', 'n_clicks')],
              [State('gw-selection-dropdown', 'value')],
              prevent_initial_call=True)
def execute_fastai_potential_training(n_clicks, gw):
    print("FastAI Potential Model click={}".format(n_clicks))
    if n_clicks:
        if not gw:
            return html.P("Please Select GW")
        print("Training FastAI Potential Model for gameweek={}".format(gw))
        loss_history = perform_fastai_potential_training(gw)
        train_loss, valid_loss = loss_history["train"], loss_history["valid"]
        iterations = [i for i in range(len(train_loss))]
        num_valid_steps = len(valid_loss)
        valid_step_size = len(iterations) / num_valid_steps
        valid_iterations = [(i + 1) * valid_step_size for i in range(num_valid_steps)]
        # print(iterations)
        train_plot = go.Scatter(x=iterations, y=train_loss, mode='lines', name='Train')
        valid_plot = go.Scatter(x=valid_iterations, y=valid_loss, mode='lines', name='Valid')
        fig = make_line_plot([train_plot, valid_plot], xlabel='# Batches Processed', ylabel='L2 Loss')
        fig.update_xaxes(range=(0, len(train_loss)), ticks="inside", tick0=0, dtick=1000)
        # fig.update_yaxes(range=(0, 10), ticks="inside", tick0=1)
        legend = dict(x=0.02, y=0.02, traceorder="normal", font=dict(family="sans-serif", size=12))
        fig.update_layout(legend=legend, title="FastAI Training History")

        graph = dcc.Graph(figure=fig)
        return graph
    else:
        return html.P("Button Not Clicked!", style={"text-align": "center"})


@app.callback(Output('fastai-xreturn-outcome', 'children'),
              [Input('fastai-xreturn-btn', 'n_clicks')],
              [State('gw-selection-dropdown', 'value')],
              prevent_initial_call=True)
def execute_fastai_return_training(n_clicks, gw):
    print("FastAI Return Model click={}".format(n_clicks))
    if n_clicks:
        if not gw:
            return html.P("Please Select GW")
        print("Training FastAI Return Model for gameweek={}".format(gw))
        loss_history = perform_fastai_return_training(gw)
        train_loss, valid_loss = loss_history["train"], loss_history["valid"]
        iterations = [i for i in range(len(train_loss))]
        num_valid_steps = len(valid_loss)
        valid_step_size = len(iterations) / num_valid_steps
        valid_iterations = [(i + 1) * valid_step_size for i in range(num_valid_steps)]
        # print(iterations)
        train_plot = go.Scatter(x=iterations, y=train_loss, mode='lines', name='Train')
        valid_plot = go.Scatter(x=valid_iterations, y=valid_loss, mode='lines', name='Valid')
        fig = make_line_plot([train_plot, valid_plot], xlabel='# Batches Processed', ylabel='Log Loss')
        fig.update_xaxes(range=(0, len(train_loss)), ticks="inside", tick0=0, dtick=1000)
        # fig.update_yaxes(range=(0, 10), ticks="inside", tick0=1)
        legend = dict(x=0.02, y=0.02, traceorder="normal", font=dict(family="sans-serif", size=12))
        fig.update_layout(legend=legend, title="FastAI Training History")

        graph = dcc.Graph(figure=fig)
        return graph
    else:
        return html.P("Button Not Clicked!", style={"text-align": "center"})


# scoring
@cache.memoize(timeout=TIMEOUT)
def perform_lgbm_point_scoring(gw):
    model, _ = perform_lgbm_point_training(gw)
    XY_train, XY_test, XY_scoring, features_dict = load_data(gw)
    preds = model.predict(XY_scoring)
    df = pd.DataFrame()
    df["player_id"] = XY_scoring["player_id"].values
    df["gw"] = gw
    df['lgbm_point_pred'] = preds
    save_path = os.path.join(model.model_output_dir, "lgbm_point_predictions_gw_{}.csv".format(gw))
    df.to_csv(save_path, index=False)
    print(df.head())
    result = html.P("Done!", style={"text-align": "center"})
    return result


@app.callback(Output('lgbm-point-predict-output', 'children'),
              [Input('lgbm-point-predict-btn', 'n_clicks')],
              [State('gw-selection-dropdown', 'value')],
              prevent_initial_call=True)
def execute_lgbm_point_scoring(n_clicks, gw):
    print("LGBM Point Prediction click={}".format(n_clicks))
    if n_clicks:
        if not gw:
            return html.P("Please Select GW")
        print("Scoring from LGBM Point Model for gameweek={}".format(gw))
        result = perform_lgbm_point_scoring(gw)
        return result
    else:
        return html.P("Button Not Clicked!", style={"text-align": "center"})


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


@app.callback(Output('fastai-point-predict-output', 'children'),
              [Input('fastai-point-predict-btn', 'n_clicks')],
              [State('gw-selection-dropdown', 'value')],
              prevent_initial_call=True)
def execute_fastai_point_scoring(n_clicks, gw):
    print("FastAI Point Prediction click={}".format(n_clicks))
    if n_clicks:
        if not gw:
            return html.P("Please Select GW")
        print("Scoring from FastAI Point Model for gameweek={}".format(gw))
        result = perform_fastai_point_scoring(gw)
        return html.Div("Done!", style={"text-align": "center"})
    else:
        return html.Div("Button Not Clicked!", style={"text-align": "center"})
