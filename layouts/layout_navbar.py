import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


def get_navigation_tabs():
    tabs = dbc.Tabs(
        id="navigation-tab",
        active_tab='Home',
        children=[
            dbc.Tab(label='Home', tab_id='Home'),
            dbc.Tab(label='Leads', tab_id='Leads'),
            dbc.Tab(label='Squad', tab_id='Squad'),
            dbc.Tab(label='League', tab_id='League')
        ],
        className="tabs-modifier",
    )

    return tabs


def get_navigation_bar():
    tabs = get_navigation_tabs()
    nav_bar = dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        html.Img(src="/assets/fpl-logo.jpg", height="60px"),
                        dbc.NavbarBrand(" FPL Predictor ", className="ml-2 title-modifier"),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                dbc.Col([tabs], className='ml-auto', width='auto'),
            ]
        ),
        color="dark",
        dark=True,
        className="mb-5 banner-modifier",
    )

    nav_bar = dbc.Navbar(
        children=[
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/assets/fpl-logo.jpg", height="60px")),
                        dbc.NavbarBrand(" FPL Predictor ", className="ml-2 title-modifier"),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                dbc.Col([tabs], className='ml-auto', width='auto'),
            ],
        color="dark",
        dark=True,
        className="mb-5 banner-modifier",
        )

    return nav_bar
