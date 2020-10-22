import dash
import dash_core_components as dcc 
import dash_html_components as html 
import dash_bootstrap_components as dbc

def get_navigation_items():
    
    dropdown = dbc.DropdownMenu(
        label="Explore",
        children=[
            dbc.DropdownMenuItem("Home", href="/home"),
            dbc.DropdownMenuItem("FPL Leads", href="/lgbm-ml"),
            dbc.DropdownMenuItem("Squad Selection", href="/squad-optimizer"),
            dbc.DropdownMenuItem("League Stats", href="/classic-league"),
        ],
        nav=True,
        in_navbar=True, # set this true if dropdown menu is inside a navbar
    )
    return dropdown

def get_navigation_bar():
    
    dropdown = get_navigation_items()
    navbar = dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src="/assets/fpl-logo.jpg", height="30px")),
                            dbc.Col(dbc.NavbarBrand("FPL Predictor", className="ml-2")),
                        ],
                        align="center",
                        no_gutters=True,
                    ),
                    href="/home",
                ),
                # dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav(
                        [dropdown], className="ml-auto", navbar=True
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ]
        ),
        color="dark",
        dark=True,
        className="mb-4",
    ) 
    return navbar
