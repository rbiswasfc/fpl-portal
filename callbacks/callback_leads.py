import os
import pdb
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
    from scripts.data_loader import DataLoader
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

TIMEOUT = 3600 * 12


def load_dataframe(path):
    try:
        df = pd.read_csv(path)
    except:
        print("Error in reading {}".format(path))
        return pd.DataFrame()
    return df


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
        train_loss, valid_loss = evaluation_results["training"]["auc"], evaluation_results["valid_1"]["auc"]
        iterations = [i for i in range(len(train_loss))]
        # print(iterations)
        train_plot = go.Scatter(x=iterations, y=train_loss, mode='lines', name='Train')
        valid_plot = go.Scatter(x=iterations, y=valid_loss, mode='lines', name='Valid')
        fig = make_line_plot([train_plot, valid_plot], xlabel='# Iterations', ylabel='AUC')
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


@app.callback(Output('lgbm-potential-predict-output', 'children'),
              [Input('lgbm-potential-predict-btn', 'n_clicks')],
              [State('gw-selection-dropdown', 'value')],
              prevent_initial_call=True)
def execute_lgbm_potential_scoring(n_clicks, gw):
    print("LGBM Potential Prediction click={}".format(n_clicks))
    if n_clicks:
        if not gw:
            return html.P("Please Select GW")
        print("Scoring from LGBM Potential Model for gameweek={}".format(gw))
        result = perform_lgbm_potential_scoring(gw)
        return result
    else:
        return html.P("Button Not Clicked!", style={"text-align": "center"})


@app.callback(Output('lgbm-return-predict-output', 'children'),
              [Input('lgbm-return-predict-btn', 'n_clicks')],
              [State('gw-selection-dropdown', 'value')],
              prevent_initial_call=True)
def execute_lgbm_return_scoring(n_clicks, gw):
    print("LGBM Return Prediction click={}".format(n_clicks))
    if n_clicks:
        if not gw:
            return html.P("Please Select GW")
        print("Scoring from LGBM Return Model for gameweek={}".format(gw))
        result = perform_lgbm_return_scoring(gw)
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


@app.callback(Output('fastai-potential-predict-output', 'children'),
              [Input('fastai-potential-predict-btn', 'n_clicks')],
              [State('gw-selection-dropdown', 'value')],
              prevent_initial_call=True)
def execute_fastai_potential_scoring(n_clicks, gw):
    print("FastAI Potential Prediction click={}".format(n_clicks))
    if n_clicks:
        if not gw:
            return html.P("Please Select GW")
        print("Scoring from FastAI Potential Model for gameweek={}".format(gw))
        result = perform_fastai_potential_scoring(gw)
        return html.Div("Done!", style={"text-align": "center"})
    else:
        return html.Div("Button Not Clicked!", style={"text-align": "center"})


@app.callback(Output('fastai-return-predict-output', 'children'),
              [Input('fastai-return-predict-btn', 'n_clicks')],
              [State('gw-selection-dropdown', 'value')],
              prevent_initial_call=True)
def execute_fastai_return_scoring(n_clicks, gw):
    print("FastAI Return Prediction click={}".format(n_clicks))
    if n_clicks:
        if not gw:
            return html.P("Please Select GW")
        print("Scoring from FastAI Potential Model for gameweek={}".format(gw))
        result = perform_fastai_return_scoring(gw)
        return html.Div("Done!", style={"text-align": "center"})
    else:
        return html.Div("Button Not Clicked!", style={"text-align": "center"})


# Leads Update
@app.callback([Output('gk-leads', 'children'),
               Output('def-leads', 'children'),
               Output('mid-leads', 'children'),
               Output('fwd-leads', 'children'), ],
              [Input('team-selection-dropdown-leads', 'value'),
               Input('model-selection-dropdown-leads', 'value'),
               Input('gw-selection-dropdown', 'value')],
              prevent_initial_call=True)
def execute_fastai_return_scoring(team_name, model_name, gw_id):
    if not gw_id:
        msg = html.P("Please select GW for scoring")
        return msg, msg, msg, msg

    if not model_name:
        msg = html.P("Please select Model")
        return msg, msg, msg, msg

    model_name_col_map = {
        "LGBM Point": "lgbm_point_pred",
        "LGBM Potential": "lgbm_potential_pred",
        "LGBM Return": "lgbm_return_pred",
        "Fast Point": "fastai_point_pred",
        "Fast Potential": "fastai_potential_pred",
        "Fast Return": "fastai_return_pred"
    }

    print("Leads for {} in gw {}".format(team_name, gw_id))
    data_maker = ModelDataMaker(CONFIG_2020)
    output_dir = "./data/model_outputs/"

    # load model predictions
    lgbm_point_path = os.path.join(output_dir, "lgbm_point_predictions_gw_{}.csv".format(gw_id))
    lgbm_potential_path = os.path.join(output_dir, "lgbm_potential_predictions_gw_{}.csv".format(gw_id))
    lgbm_return_path = os.path.join(output_dir, "lgbm_return_predictions_gw_{}.csv".format(gw_id))

    fastai_point_path = os.path.join(output_dir, "fastai_point_predictions_gw_{}.csv".format(gw_id))
    fastai_potential_path = os.path.join(output_dir, "fastai_potential_predictions_gw_{}.csv".format(gw_id))
    fastai_return_path = os.path.join(output_dir, "fastai_return_predictions_gw_{}.csv".format(gw_id))

    df_lgbm_point = load_dataframe(lgbm_point_path)
    df_lgbm_potential = load_dataframe(lgbm_potential_path)
    df_lgbm_return = load_dataframe(lgbm_return_path)
    df_fastai_point = load_dataframe(fastai_point_path)
    df_fastai_potential = load_dataframe(fastai_potential_path)
    df_fastai_return = load_dataframe(fastai_return_path)

    all_preds_df = [df_lgbm_point, df_lgbm_potential, df_lgbm_return,
                    df_fastai_point, df_fastai_potential, df_fastai_return]

    for df in all_preds_df:
        try:
            assert len(df) > 0
        except:
            msg = html.P("Run scoring for models before generating leads")
            return msg, msg, msg, msg

    # prepare prediction base dataframe
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

    if team_name != "All":
        df_leads = df_leads[df_leads["team"] == team_name].copy()

    # merge predictions
    for df in all_preds_df:
        df = df.drop_duplicates()
        df_leads = pd.merge(df_leads, df, how='left', on=['player_id', 'gw'])
    # keep_cols = ["name", "cost", "position", "selection_pct", "next_opponent", "lgbm_point_pred", "lgbm_potential_pred"]
    # df_leads = df_leads[keep_cols].copy()
    # make tables
    df_leads["cost"] = df_leads["cost"] / 10
    model_col = model_name_col_map[model_name]
    df_leads = df_leads.sort_values(by=model_col, ascending=False)

    # column round up
    pred_cols = ["lgbm_point_pred", "lgbm_potential_pred", "lgbm_return_pred",
                 "fastai_point_pred", "fastai_potential_pred", "fastai_return_pred"]
    for col in pred_cols:
        df_leads[col] = df_leads[col].round(2)

    df_gk = df_leads[df_leads["position"] == "GK"].copy()
    df_def = df_leads[df_leads["position"] == "DEF"].copy()
    df_mid = df_leads[df_leads["position"] == "MID"].copy()
    df_fwd = df_leads[df_leads["position"] == "FWD"].copy()

    col_map = {"name": "Player", "cost": "Cost", "next_opponent": "Opponent",
               "selection_pct": "TSB"}
    base_cols = ["name", "cost", "selection_pct", "next_opponent", model_col]
    col_map[model_col] = model_name

    df_gk = df_gk[base_cols].copy()
    df_gk = df_gk.rename(columns=col_map)
    gk_table = make_table(df_gk)

    df_def = df_def[base_cols].copy()
    df_def = df_def.rename(columns=col_map)
    def_table = make_table(df_def)

    df_mid = df_mid[base_cols].copy()
    df_mid = df_mid.rename(columns=col_map)
    mid_table = make_table(df_mid)

    df_fwd = df_fwd[base_cols].copy()
    df_fwd = df_fwd.rename(columns=col_map)
    fwd_table = make_table(df_fwd)
    return gk_table, def_table, mid_table, fwd_table


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


@app.callback(Output('shap-output', 'children'),
              [Input('player-selection-dropdown-shap', 'value'),
               Input('model-selection-dropdown-shap', 'value'),
               Input('gw-selection-dropdown', 'value')],
              prevent_initial_call=True)
def execute_shap_explanation(player_name, model_name, gw_id):
    if not gw_id:
        msg = html.P("Please select GW for scoring")
        return msg

    if not player_name:
        msg = html.P("Please select Player")
        return msg

    if not model_name:
        msg = html.P("Please select Model")
        return msg

    # get shap dataframe
    df_shap, ave_score = perform_shap_analysis(model_name, gw_id)
    print(ave_score)

    df_this_player = df_shap[df_shap["name"] == player_name].copy()
    df_this_player = df_this_player.drop_duplicates(subset=["name"])

    if len(df_this_player) == 0:
        return html.P("Player Not Found")
    shap_cols = [col for col in df_this_player.columns if col.startswith('shap')]
    feature_cols = [col.split("shap_")[1] for col in shap_cols]
    explanation_list = []
    for feat, feat_shap in zip(feature_cols, shap_cols):
        feat_val = df_this_player[feat].values[0]
        shap_val = df_this_player[feat_shap].values[0]
        this_exp = {"feature": feat, "feature_val": feat_val, "shap_val": shap_val}
        explanation_list.append(this_exp)

    df_meta = pd.DataFrame(explanation_list)
    df_meta["abs_shap"] = df_meta["shap_val"].apply(lambda x: abs(x))
    df_meta = df_meta.sort_values(by="abs_shap", ascending=False)
    score = df_meta["shap_val"].sum() + ave_score
    df_top_exp = df_meta.iloc[:20]

    fig = px.bar(df_top_exp, y='shap_val', x='feature', text='feature_val', title="{}: Score = {:.2f}".format(player_name, score),
                 labels={"shap_val": "SHAP", "feature": "Feature"},
                 template="seaborn")

    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_layout(margin={'l': 5, 'b': 75, 't': 25, 'r': 0})
    imp_bar = dcc.Graph(figure=fig)

    return imp_bar


