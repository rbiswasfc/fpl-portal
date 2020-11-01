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

TIMEOUT = 3600


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
            html.Div(make_button("INGEST", button_ids[0]), className="col-4"),
            dcc.Loading(html.Div(id="data-ingest-div", className="col-4"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    button_fe = html.Div(
        children=[
            html.Div(make_button("Feature Engineering", button_ids[1]), className="col-4"),
            dcc.Loading(html.Div(id="data-fe-div", className="col-4"), color='black'),
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

            html.Div("Points Predictor", className='subtitle inline-header'),

            html.Div("Scoring", className='subtitle inline-header'),
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
