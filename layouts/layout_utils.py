import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.io as plt_io


def make_header(text):
    section = html.P(
        className="section-title",
        children=text
    )
    return section


def make_table(df):
    table = dash_table.DataTable(
        columns=[{'name': col, 'id': col} for col in df.columns],
        data=df.to_dict('records'),
        # filter_action='native',
        style_as_list_view=True,
        style_header={'backgroundColor': 'rgb(30, 30, 30)'},
        style_cell={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'white',
            'textAlign': 'center'
        },
        sort_action='native',
        page_size=6
    )
    layout = html.Div(table, className='container', style={"width": "95%"})
    return layout


def make_dropdown(dropdown_id, dropdown_options, placeholder=None, multi_flag=False):
    if dropdown_options:
        dropdown = dcc.Dropdown(
            id=dropdown_id,
            options=dropdown_options,
            placeholder=placeholder,
            multi=multi_flag
        )
    else:
        dropdown = dcc.Dropdown(
            id=dropdown_id,
            placeholder=placeholder,
            multi=multi_flag
        )

    dropdown_section = dbc.Row(
        children=[
            html.Div(dropdown, className='col-12')
        ],
        style={'margin-top': '1rem', 'margin-bottom': '2rem'}
    )
    return dropdown_section


def make_line_plot(data, xlabel=None, ylabel=None):

    layout = go.Layout(xaxis={'title': xlabel},
                       yaxis={'title': ylabel},
                       margin={'l': 5, 'b': 75, 't': 25, 'r': 5},
                       hovermode='x')
    figure = go.Figure(data=data, layout=layout)
    figure.layout.template = 'seaborn'

    return figure
