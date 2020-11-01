import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from flask_caching import Cache

try:
    from layouts.layout_utils import make_header, make_table, make_dropdown, make_button
    from scripts.data_loader import DataLoader
    from scripts.data_scrape import DataScraper
    from scripts.utils import load_config
    from app import cache
except:
    raise ImportError

TIMEOUT = 3600 * 5


@cache.memoize(timeout=TIMEOUT)
def query_next_gameweek():
    config = load_config()
    data_loader = DataLoader(config)
    next_gw = int(data_loader.get_next_gameweek_id())
    return next_gw


def make_gw_selection_section():
    next_gw = query_next_gameweek()
    focus_gws = [next_gw - 2, next_gw - 1, next_gw]
    gw_options = [{'label': gw_id, 'value': gw_id} for gw_id in focus_gws]
    dropdown_section = make_dropdown('gw-selection-dropdown', gw_options,
                                     placeholder="Select Gameweek ID ...")
    return dropdown_section


def make_pipeline_section():
    button_ids = ["data-ingest-btn", "data-fe-btn"]
    margin_style = {"margin-top": "1rem", "margin-bottom": "1rem"}

    button_ingest = html.Div(
        children=[
            html.Div(make_button("INGEST", button_ids[0]), className="col-6"),
            dcc.Loading(html.Div(id="data-ingest-div", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    button_fe = html.Div(
        children=[
            html.Div(make_button("Feature Engineering", button_ids[1]), className="col-6"),
            dcc.Loading(html.Div(id="data-fe-div", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    pipeline = html.Div(
        children=[
            html.Div("Data Pipeline", className='subtitle inline-header'),
            button_ingest,
            button_fe
        ])
    pipeline_section = html.Div(pipeline)
    return pipeline_section


def make_points_predictor_section():
    margin_style = {"margin-top": "1rem", "margin-bottom": "2rem"}

    button_train = html.Div(
        children=[
            html.Div(make_button("TRAIN LGB MODEL", 'lgbm-xnext-btn'), className="col-6"),
            html.Div(make_button("TRAIN FastAI MODEL", 'fastai-xnext-btn'), className="col-6"),
        ],
        className="row",
        style=margin_style
    )

    train_output = html.Div(
        children=[
            dcc.Loading(html.Div(id="lgbm-xnext-outcome", className='six columns', style={"width": "80%"}),
                        color='black'),
            dcc.Loading(html.Div(id="fastai-xnext-outcome", className='six columns', style={"width": "50%"}),
                        color='black'),
        ],
        className="row",
        style=margin_style
    )

    point_predictor = html.Div(
        children=[
            html.Div("Points Predictor", className='subtitle inline-header'),
            button_train,
            # train_output,
            dcc.Loading(html.Div(id="lgbm-xnext-outcome"), color='black'),
            dcc.Loading(html.Div(id="fastai-xnext-outcome"), color='black'),
        ])
    point_predictor_section = html.Div(point_predictor)
    return point_predictor_section


def make_scoring_section():
    margin_style = {"margin-top": "1rem", "margin-bottom": "2rem"}

    button_scoring_lgbm_point = html.Div(
        children=[
            html.Div(make_button("LGBM Point Predictor", 'lgbm-point-predict-btn'), className="col-6"),
            dcc.Loading(html.Div(id="lgbm-point-predict-output", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    button_scoring_fastai_point = html.Div(
        children=[
            html.Div(make_button("FastAI Point Predictor", 'fastai-point-predict-btn'), className="col-6"),
            dcc.Loading(html.Div(id="fastai-point-predict-output", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    button_scoring_lgbm_potential = html.Div(
        children=[
            html.Div(make_button("LGBM Potential Predictor", 'lgbm-potential-predict-btn'), className="col-6"),
            dcc.Loading(html.Div(id="lgbm-potential-predict-output", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    button_scoring_fastai_potential = html.Div(
        children=[
            html.Div(make_button("FastAI Potential Predictor", 'fastai-potential-predict-btn'), className="col-6"),
            dcc.Loading(html.Div(id="fastai-potential-predict-output", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    button_scoring_lgbm_return = html.Div(
        children=[
            html.Div(make_button("LGBM Return Predictor", 'lgbm-return-predict-btn'), className="col-6"),
            dcc.Loading(html.Div(id="lgbm-return-predict-output", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    button_scoring_fastai_return = html.Div(
        children=[
            html.Div(make_button("FastAI Return Predictor", 'fastai-return-predict-btn'), className="col-6"),
            dcc.Loading(html.Div(id="fastai-return-predict-output", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    scoring = html.Div(
        children=[
            html.Div("Scoring", className='subtitle inline-header'),
            button_scoring_lgbm_point,
            button_scoring_fastai_point,
            button_scoring_lgbm_potential,
            button_scoring_fastai_potential,
            button_scoring_lgbm_return,
            button_scoring_fastai_return
        ])
    scoring_section = html.Div(scoring)
    return scoring_section


def make_left_layout_leads():
    header = make_header("Model Training")
    layout = html.Div(
        id='leads-layout-left',
        className="six columns",
        children=[
            header,
            html.Div("Select Gameweek", className='subtitle inline-header'),
            make_gw_selection_section(),
            make_pipeline_section(),
            make_points_predictor_section(),
            make_scoring_section(),
        ],
    )
    return layout


def make_right_layout_leads():
    header = make_header("Generated Leads")
    layout = html.Div(
        id='leads-layout-left',
        className="six columns",
        children=[
            header,
            html.Div("GK Leads", className='subtitle inline-header'),
            html.Div("DEF Leads", className='subtitle inline-header'),
            html.Div("MID Leads", className='subtitle inline-header'),
            html.Div("FWD Leads", className='subtitle inline-header'),
        ],
    )
    return layout


def make_leads_page():
    left_layout = make_left_layout_leads()
    right_layout = make_right_layout_leads()
    layout = html.Div(
        children=[
            left_layout,
            right_layout
        ]
    )
    return layout


if __name__ == "__main__":
    pass
