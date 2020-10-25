import dash
import dash_core_components as dcc 
import dash_html_components as html
import dash_bootstrap_components as dbc

# bootstrap theme
# https://bootswatch.com/lux/
external_stylesheets = [dbc.themes.BOOTSTRAP]


app = dash.Dash(__name__, 
                external_stylesheets=external_stylesheets,
                assets_folder='assets/',
                )

app.config.suppress_callback_exceptions = True
app.title = "FPL Buddy"

app.layout = html.Div(children=[
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', style={"height": "100%", "width": "100%"}),
])

server = app.server

