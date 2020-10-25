import dash 
import dash_core_components as dcc 
import dash_html_components as html 
import dash_bootstrap_components as dbc 
from layouts.layout_utils import make_header

def make_left_layout_league():
    header = make_header("Classic League Analytics")
    layout =  html.Div(
                id = 'league-layout-left',
                className = "six columns",
                children = [
                    header,
                    html.Div("This is standings")
                    ],
                )
    return layout

def make_right_layout_league():
    pass

def make_league_page():
    left_layout = make_left_layout_league()
    layout = html.Div(
        children = [
            left_layout,
            left_layout],
        )
    return layout 

if __name__ == "__main__":
    pass 
