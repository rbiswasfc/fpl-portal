import dash
import dash_html_components as html
from dash.dependencies import Input, Output
from app import app
from layouts.layout_league import make_league_page
from layouts.layout_leads import make_leads_page
from layouts.layout_squad import make_squad_page


@app.callback(Output('current-tab-content', 'children'),
              [Input('navigation-tab', 'active_tab')])
def render_content(tab):
    print("current tab = {}".format(tab))
    if tab == 'Home':
        return html.Div("This is HOME Page")
    elif tab == 'Leads':
        return make_leads_page()
    elif tab == 'Squad':
        return make_squad_page()
    elif tab == 'League':
        return make_league_page()
    else:
        return html.Div("OH BOY! This is 404!")
