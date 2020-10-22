import dash
import dash_core_components as dcc 
import dash_html_components as html
import dash_bootstrap_components as dbc

from navbar import get_navigation_bar

# bootstrap theme
# https://bootswatch.com/lux/
external_stylesheets = [dbc.themes.LUX]


app = dash.Dash(__name__, 
                external_stylesheets=external_stylesheets,
                assets_folder='assets/',
                )

app.config.suppress_callback_exceptions = True
app.title = "FPL AIXI"

app_navbar = get_navigation_bar()
app.layout = html.Div(children=[
    dcc.Location(id='url', refresh=False),
    app_navbar,
    html.Div(id='page-content', style={"height": "100%", "width": "100%"}),
])

server = app.server

