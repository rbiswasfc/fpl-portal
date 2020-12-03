import dash_core_components as dcc
import dash_html_components as html

try:
    from layouts.layout_utils import make_header, make_table, make_dropdown, make_button
    from scripts.data_loader import DataLoader
    from scripts.data_scrape import DataScraper
    from scripts.data_preparation import ModelDataMaker
    from scripts.utils import load_config
    from layouts.layout_cache import query_next_gameweek, CONFIG_2020
except:
    raise ImportError


def make_gw_selection_section():
    next_gw = query_next_gameweek()
    focus_gws = [next_gw - 2, next_gw - 1, next_gw]
    gw_options = [{'label': gw_id, 'value': gw_id} for gw_id in focus_gws]
    dropdown_section = make_dropdown('gw-selection-dropdown', gw_options,
                                     placeholder="Select Gameweek ID ...")
    return dropdown_section


def make_pipeline_section():
    button_ids = ["data-ingest-btn", "data-fe-btn"]
    margin_style = {"margin-top": "1rem", "margin-bottom": "1rem"}

    button_ingest = html.Div(
        children=[
            html.Div(make_button("INGEST", button_ids[0]), className="col-6"),
            dcc.Loading(html.Div(id="data-ingest-div", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    button_fe = html.Div(
        children=[
            html.Div(make_button("Feature Engineering", button_ids[1]), className="col-6"),
            dcc.Loading(html.Div(id="data-fe-div", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    pipeline = html.Div(
        children=[
            html.Div("Data Pipeline", className='subtitle inline-header'),
            button_ingest,
            button_fe
        ])
    pipeline_section = html.Div(pipeline)
    return pipeline_section


def make_points_predictor_section():
    margin_style = {"margin-top": "1rem", "margin-bottom": "2rem"}

    button_train = html.Div(
        children=[
            html.Div(make_button("TRAIN LGB MODEL", 'lgbm-xnext-btn'), className="col-6"),
            html.Div(make_button("TRAIN FastAI MODEL", 'fastai-xnext-btn'), className="col-6"),
        ],
        className="row",
        style=margin_style
    )

    point_predictor = html.Div(
        children=[
            html.Div("Points Predictor", className='subtitle inline-header'),
            button_train,
            dcc.Loading(html.Div(id="lgbm-xnext-outcome"), color='black'),
            dcc.Loading(html.Div(id="fastai-xnext-outcome"), color='black'),
            dcc.Loading(html.Div(id="xnext-feature-imp", style={"width": "100%"}), color='black'),
        ])
    point_predictor_section = html.Div(point_predictor)
    return point_predictor_section


def make_return_predictor_section():
    margin_style = {"margin-top": "1rem", "margin-bottom": "2rem"}

    button_train = html.Div(
        children=[
            html.Div(make_button("TRAIN LGB MODEL", 'lgbm-xreturn-btn'), className="col-6"),
            html.Div(make_button("TRAIN FastAI MODEL", 'fastai-xreturn-btn'), className="col-6"),
        ],
        className="row",
        style=margin_style
    )

    return_predictor = html.Div(
        children=[
            html.Div("Return Predictor", className='subtitle inline-header'),
            button_train,
            # train_output,
            dcc.Loading(html.Div(id="lgbm-xreturn-outcome"), color='black'),
            dcc.Loading(html.Div(id="fastai-xreturn-outcome"), color='black'),
            dcc.Loading(html.Div(id="xreturn-feature-imp", style={"width": "100%"}), color='black'),
        ])
    return_predictor_section = html.Div(return_predictor)
    return return_predictor_section


def make_potential_predictor_section():
    margin_style = {"margin-top": "1rem", "margin-bottom": "2rem"}

    button_train = html.Div(
        children=[
            html.Div(make_button("TRAIN LGB MODEL", 'lgbm-xpotential-btn'), className="col-6"),
            html.Div(make_button("TRAIN FastAI MODEL", 'fastai-xpotential-btn'), className="col-6"),
        ],
        className="row",
        style=margin_style
    )

    potential_predictor = html.Div(
        children=[
            html.Div("Potential Predictor", className='subtitle inline-header'),
            button_train,
            dcc.Loading(html.Div(id="lgbm-xpotential-outcome"), color='black'),
            dcc.Loading(html.Div(id="fastai-xpotential-outcome"), color='black'),
            dcc.Loading(html.Div(id="xpotential-feature-imp", style={"width": "100%"}), color='black'),
        ])
    potential_predictor_section = html.Div(potential_predictor)
    return potential_predictor_section


def make_scoring_section():
    margin_style = {"margin-top": "1rem", "margin-bottom": "2rem"}

    button_scoring_lgbm_point = html.Div(
        children=[
            html.Div(make_button("LGBM Point Predictor", 'lgbm-point-predict-btn'), className="col-6"),
            dcc.Loading(html.Div(id="lgbm-point-predict-output", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    button_scoring_fastai_point = html.Div(
        children=[
            html.Div(make_button("FastAI Point Predictor", 'fastai-point-predict-btn'), className="col-6"),
            dcc.Loading(html.Div(id="fastai-point-predict-output", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    button_scoring_lgbm_potential = html.Div(
        children=[
            html.Div(make_button("LGBM Potential Predictor", 'lgbm-potential-predict-btn'), className="col-6"),
            dcc.Loading(html.Div(id="lgbm-potential-predict-output", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    button_scoring_fastai_potential = html.Div(
        children=[
            html.Div(make_button("FastAI Potential Predictor", 'fastai-potential-predict-btn'), className="col-6"),
            dcc.Loading(html.Div(id="fastai-potential-predict-output", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    button_scoring_lgbm_return = html.Div(
        children=[
            html.Div(make_button("LGBM Return Predictor", 'lgbm-return-predict-btn'), className="col-6"),
            dcc.Loading(html.Div(id="lgbm-return-predict-output", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    button_scoring_fastai_return = html.Div(
        children=[
            html.Div(make_button("FastAI Return Predictor", 'fastai-return-predict-btn'), className="col-6"),
            dcc.Loading(html.Div(id="fastai-return-predict-output", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    button_scoring_lgbm_future = html.Div(
        children=[
            html.Div(make_button("LGBM Next GWs Predictor", 'lgbm-next-7-predict-btn'), className="col-6"),
            dcc.Loading(html.Div(id="lgbm-next-7-predict-output", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    button_scoring_fdr = html.Div(
        children=[
            html.Div(make_button("FDR Predictor", 'lgbm-fdr-predict-btn'), className="col-6"),
            dcc.Loading(html.Div(id="lgbm-fdr-predict-output", className="col-6"), color='black'),
        ],
        className="row",
        style=margin_style
    )

    scoring = html.Div(
        children=[
            html.Div("Scoring", className='subtitle inline-header'),
            button_scoring_lgbm_point,
            button_scoring_fastai_point,
            button_scoring_lgbm_potential,
            button_scoring_fastai_potential,
            button_scoring_lgbm_return,
            button_scoring_fastai_return,
            button_scoring_lgbm_future,
            button_scoring_fdr
        ])
    scoring_section = html.Div(scoring)
    return scoring_section


def make_lead_generation_section():
    margin_style = {"margin-top": "1rem", "margin-bottom": "2rem"}
    data_maker = ModelDataMaker(CONFIG_2020)
    team_id_team_name_map = data_maker.get_team_id_team_name_map()
    team_names = []
    for k, v in team_id_team_name_map.items():
        team_names.append(v)
    team_names = list(set(team_names))
    team_names.append("All")
    team_names = sorted(team_names)

    team_options = [{'label': team, 'value': team} for team in team_names]
    dropdown_team = make_dropdown('team-selection-dropdown-leads', team_options,
                                  placeholder="Select Team ...")

    ai_models = ["LGBM Point", "LGBM Potential", "LGBM Return", "Fast Point", "Fast Potential", "Fast Return"]
    model_options = [{'label': model, 'value': model} for model in ai_models]
    dropdown_model = make_dropdown('model-selection-dropdown-leads', model_options,
                                   placeholder="Select Model ...")
    dropdown_section = html.Div(
        children=[
            html.Div(dropdown_team, className='col-6'),
            html.Div(dropdown_model, className='col-6'),
        ],
        className='row'
    )

    leads_output = html.Div(
        children=[
            html.Div("GK", className='subtitle inline-header'),
            dcc.Loading(html.Div(id='gk-leads', style=margin_style), color='black'),
            html.Div("DEF", className='subtitle inline-header'),
            dcc.Loading(html.Div(id='def-leads', style=margin_style), color='black'),
            html.Div("MID", className='subtitle inline-header'),
            dcc.Loading(html.Div(id='mid-leads', style=margin_style), color='black'),
            html.Div("FWD", className='subtitle inline-header'),
            dcc.Loading(html.Div(id='fwd-leads', style=margin_style), color='black')
        ])

    section = html.Div(
        children=[
            html.Div("Select Team & Model", className='subtitle inline-header'),
            dropdown_section,
            leads_output
        ]
    )
    return section


def make_shap_explanation_section():
    margin_style = {"margin-top": "1rem", "margin-bottom": "2rem"}
    data_maker = ModelDataMaker(CONFIG_2020)
    player_id_player_name_map = data_maker.get_player_id_player_name_map()
    player_names = []
    for k, v in player_id_player_name_map.items():
        player_names.append(v)
    player_names = sorted(list(set(player_names)))

    player_options = [{'label': player, 'value': player} for player in player_names]
    dropdown_player = make_dropdown('player-selection-dropdown-shap', player_options,
                                    placeholder="Select Player ...")

    ai_models = ["LGBM Point", "LGBM Potential", "LGBM Return"]
    model_options = [{'label': model, 'value': model} for model in ai_models]
    dropdown_model = make_dropdown('model-selection-dropdown-shap', model_options,
                                   placeholder="Select Model ...")
    dropdown_section = html.Div(
        children=[
            html.Div(dropdown_player, className='col-6'),
            html.Div(dropdown_model, className='col-6'),
        ],
        className='row'
    )

    shap_output = html.Div(
        children=[
            dcc.Loading(html.Div(id='shap-output', style=margin_style), color='black'),
        ])

    section = html.Div(
        children=[
            # html.Div("Select Player & Model", className='subtitle inline-header'),
            dropdown_section,
            shap_output
        ]
    )
    return section


def make_left_layout_leads():
    header = make_header("Model Training")
    layout = html.Div(
        id='leads-layout-left',
        className="six columns",
        children=[
            header,
            html.Div("Select Gameweek", className='subtitle inline-header'),
            make_gw_selection_section(),
            make_pipeline_section(),
            make_points_predictor_section(),
            make_potential_predictor_section(),
            make_return_predictor_section(),
            make_scoring_section(),
        ],
    )
    return layout


def make_right_layout_leads():
    header = make_header("Leads")
    layout = html.Div(
        id='leads-layout-left',
        className="six columns",
        children=[
            header,
            make_lead_generation_section(),
            html.Div("Shap Explanation", className='subtitle inline-header'),
            make_shap_explanation_section(),
        ],
    )
    return layout


def make_leads_page():
    left_layout = make_left_layout_leads()
    right_layout = make_right_layout_leads()
    layout = html.Div(
        children=[
            left_layout,
            right_layout
        ]
    )
    return layout


if __name__ == "__main__":
    pass
