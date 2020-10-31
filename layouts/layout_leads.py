import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

try:
    from layouts.layout_utils import make_header, make_table, make_dropdown
    from scripts.data_loader import DataLoader
    from scripts.utils import load_config
except:
    raise ImportError


def make_left_layout_leads():
    header = make_header("Model Training")
    layout = html.Div(
        id='leads-layout-left',
        className="six columns",
        children=[
            header,
            html.Div("Points Predictor", className='subtitle inline-header'),
            html.Div("Potential Predictor", className='subtitle inline-header'),
            html.Div("Return Predictor", className='subtitle inline-header'),
            html.Div("DONE", className='subtitle inline-header'),
        ],
    )
    return layout


def make_right_layout_leads():
    header = make_header("Leads")
    layout = html.Div(
        id='leads-layout-left',
        className="six columns",
        children=[
            header,
            html.Div("GK", className='subtitle inline-header'),
            html.Div("DEF", className='subtitle inline-header'),
            html.Div("MID", className='subtitle inline-header'),
            html.Div("FWD", className='subtitle inline-header'),
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
