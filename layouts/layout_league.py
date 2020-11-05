import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from layouts.layout_utils import make_header, make_table, make_dropdown
from scripts.data_loader import DataLoader
from scripts.utils import load_config


def make_league_search_section():
    config = load_config()
    league_ids = config["leagues"]
    league_options = [{'label': this_id, 'value': this_id} for this_id in league_ids]
    dropdown_section = make_dropdown('league-search-dropdown', league_options,
                                     placeholder="Select League ID ...")
    return dropdown_section


def make_league_comparison_section():
    pass


def make_left_layout_league():
    header = make_header("League Standing")
    dropdown_section = make_dropdown('team-selection-dropdown', None,
                                     placeholder="Select Teams For Comparison ...", multi_flag=True)
    layout = html.Div(
        id='league-layout-left',
        className="six columns",
        children=[
            header,
            html.Div("League Summary", className='subtitle inline-header'),
            make_league_search_section(),
            html.Div(id="league-standing-table", style={"width": "100%"}),
            dcc.Store(id="league-standing-memory"),
            html.Div("League Gameweek History", className='subtitle inline-header'),
            dropdown_section,
            html.Div(id="league-point-history", style={"width": "100%"})
        ],
    )
    return layout


def make_right_layout_league():
    header = make_header("League Analytics")
    dropdown_section = make_dropdown('league-team-picks-dropdown', None,
                                     placeholder="Select FPL Team ...", multi_flag=False)
    layout = html.Div(
        id='league-layout-left',
        className="six columns",
        children=[
            header,
            html.Div("Team Picks", className='subtitle inline-header'),
            dropdown_section,
            dcc.Loading(html.Div(id="league-team-picks-display"), color='black'),
        ],
    )
    return layout


def make_league_page():
    left_layout = make_left_layout_league()
    right_layout = make_right_layout_league()
    layout = html.Div(
        children=[
            left_layout,
            right_layout
        ]
    )
    return layout


if __name__ == "__main__":
    pass
