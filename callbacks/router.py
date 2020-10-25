import dash  
import dash_html_components as html 
from dash.dependencies import Input, Output 
from app import app
from layouts.layout_navbar import get_navigation_bar
from layouts.layout_league import make_league_page

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    section = html.Div(
        children=[get_navigation_bar(),
        html.Div(id='current-tab-content')
        ]
    )
    return section

