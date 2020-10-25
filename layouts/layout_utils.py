import dash 
import dash_core_components as dcc 
import dash_html_components as html 
import dash_bootstrap_components as dbc 

def make_header(text):
    section = html.P(
        className="section-title",
        children = text
        )
    return section 