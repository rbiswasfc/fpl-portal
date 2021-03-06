import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from flask_caching import Cache

try:
    from layouts.layout_utils import make_header, make_table, make_dropdown, make_button, make_input
    from scripts.data_loader import DataLoader
    from scripts.data_scrape import DataScraper
    from scripts.data_preparation import ModelDataMaker
    from scripts.utils import load_config
    from app import cache
    from layouts.layout_leads import query_next_gameweek
except:
    raise ImportError

TIMEOUT = 3600 * 48

CONFIG_2020 = {
    "data_dir": "./data/model_data/2020_21/",
    "file_fixture": "fixtures.csv",
    "file_team": "teams.csv",
    "file_gw": "merged_gw.csv",
    "file_player": "players_raw.csv",
    "file_understat_team": "understat_team_data.pkl",
    "scoring_gw": "NA"
}


def make_optimization_settings_section():
    margin_style = {"margin-top": "1rem", "margin-bottom": "2rem"}
    next_gw = query_next_gameweek()
    focus_gws = [next_gw - 2, next_gw - 1, next_gw]
    gw_options = [{'label': gw_id, 'value': gw_id} for gw_id in focus_gws]
    dropdown_gw = make_dropdown('gw-selection-dropdown-squad', gw_options,
                                placeholder="Select Gameweek ...")

    ai_models = ["LGBM Point", "LGBM Potential", "LGBM Return", "Fast Point", "Fast Potential", "Fast Return", "Net"]
    model_options = [{'label': model, 'value': model} for model in ai_models]
    dropdown_model = make_dropdown('model-selection-dropdown-optim', model_options,
                                   placeholder="Select Model ...")

    formations = ['1-3-4-3', '1-3-5-2', '1-4-4-2', '1-4-3-3']
    formation_options = [{'label': this_formation, 'value': this_formation} for this_formation in formations]
    dropdown_formation = make_dropdown('formation-selection-dropdown-squad', formation_options,
                                       placeholder="Select Formation ...")
    dropdown_section = html.Div(
        children=[
            html.Div(dropdown_gw, className='col-4'),
            html.Div(dropdown_model, className='col-4'),
            html.Div(dropdown_formation, className='col-4'),
        ],
        className='row'
    )

    budget_header = html.Div(
        children=[
            html.Div("Enter Squad Value", className='col-4'),
            html.Div("Enter Bench Value", className='col-4'),
            html.Div("Exclude Uncertain Players", className='col-4'),
        ],
        className='row'
    )

    budget_section = html.Div(
        children=[
            html.Div(make_input('squad-value-input', 'number', 100), className='col-4'),
            html.Div(make_input('bench-value-input', 'number', 18), className='col-4'),
            html.Div(make_input('uncertain-flag', 'text', "Yes"), className='col-4'),
        ],
        className='row'
    )

    section = html.Div(
        children=[
            dropdown_section,
            budget_header,
            budget_section,
            html.Div(make_button("Submit", 'squad-optimization-btn')),
            html.Div(" ", style={"margin-top": "6rem", "margin-bottom": "6rem"})
        ]
    )
    return section


def make_left_layout_squad():
    header = make_header("Squad Optimization")
    layout = html.Div(
        id='squad-layout-left',
        className="six columns",
        children=[
            header,
            html.Div("Settings", className='subtitle inline-header'),
            make_optimization_settings_section(),

            html.Div("Playing XI", className='subtitle inline-header'),
            html.Div(" ", style={"margin-top": "2rem", "margin-bottom": "2rem"}),
            dcc.Loading(html.Div(id='squad-optim-output-play-xi'), color='black'),

            html.Div(" ", style={"margin-top": "2rem", "margin-bottom": "2rem"}),
            html.Div("Bench", className='subtitle inline-header'),
            html.Div(" ", style={"margin-top": "2rem", "margin-bottom": "2rem"}),
            dcc.Loading(html.Div(id='squad-optim-output-bench'), color='black'),

        ],
    )
    return layout


def make_player_comparison_section():
    margin_style = {"margin-top": "1rem", "margin-bottom": "2rem"}
    data_maker = ModelDataMaker(CONFIG_2020)
    player_id_player_name_map = data_maker.get_player_id_player_name_map()
    player_names = []
    for k, v in player_id_player_name_map.items():
        player_names.append(v)
    player_names = sorted(list(set(player_names)))
    player_options = [{'label': player, 'value': player} for player in player_names]
    dropdown_player_a = make_dropdown('player-selection-dropdown-a', player_options,
                                      placeholder="Select Player ...")
    dropdown_player_b = make_dropdown('player-selection-dropdown-b', player_options,
                                      placeholder="Select Player ...")

    player_dropdown_section = html.Div(
        children=[
            html.Div(dropdown_player_a, className='col-6'),
            html.Div(dropdown_player_b, className='col-6'),
        ],
        className='row'
    )
    section = html.Div(
        children=[
            html.Div("Player Comparison", className='subtitle inline-header'),
            player_dropdown_section,
            html.Div(id='player-compare-output', style=margin_style)
        ])
    return section


def make_transfers_section():
    margin_style = {"margin-top": "1rem", "margin-bottom": "2rem"}
    config = load_config()
    data_loader = DataLoader(config)
    df_league = data_loader.get_league_standings()
    manager_ids = df_league["entry_id"].unique().tolist()
    manager_names = df_league["entry_name"].unique().tolist()
    manager_options = [{'label': manager, 'value': manager_id} for manager, manager_id in
                       zip(manager_names, manager_ids)]
    dropdown_manager = make_dropdown('manager-selection-transfers', manager_options,
                                     placeholder="Select Manager ...")
    num_transfers = [1, 2, 3, 4]
    transfer_options = [{'label': num, 'value': num} for num in num_transfers]
    dropdown_num_transfers = make_dropdown('transfer-selection-numbers', transfer_options,
                                           placeholder="Select number of transfers ...")

    dropdown_section = html.Div(
        children=[
            html.Div(dropdown_manager, className='col-6'),
            html.Div(dropdown_num_transfers, className='col-6'),
        ],
        className='row'
    )
    section = html.Div(
        children=[
            html.Div("Transfer Suggestion", className='subtitle inline-header'),
            dropdown_section,
            html.Div(make_button("Submit", 'transfer-optimization-btn')),
            dcc.Loading(html.Div(id='transfer-suggestion-output', style=margin_style), color='black')
        ])
    return section


def make_right_layout_squad():
    header = make_header("Transfers")
    layout = html.Div(
        id='squad-layout-left',
        className="six columns",
        children=[
            header,
            make_player_comparison_section(),
            make_transfers_section(),
        ],
    )
    return layout


def make_squad_page():
    left_layout = make_left_layout_squad()
    right_layout = make_right_layout_squad()
    layout = html.Div(
        children=[
            left_layout,
            right_layout
        ]
    )
    return layout


if __name__ == "__main__":
    pass
