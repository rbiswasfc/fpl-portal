import dash  
import dash_html_components as html 
from dash.dependencies import Input, Output 
from app import app

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    print("path = {}".format(pathname))
    if pathname == '/home':
        return html.Div("This is HOME Page")
    elif pathname == '/lgbm-ml':
        return html.Div("This is LEADS Page")
    elif pathname == '/squad-optimizer':
        return html.Div("This is SQUADS Page")
    elif pathname == '/classic-league':
        return html.Div("This is LEAGUES Page")
    else:
        return html.Div("OH BOY! This is 404!")