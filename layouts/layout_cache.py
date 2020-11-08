import os

try:
    from scripts.data_loader import DataLoader
    from scripts.data_preparation import ModelDataMaker
    from scripts.utils import load_config
    from app import cache
except:
    raise ImportError

TIMEOUT = 3600 * 48
TIMEOUT_SHORT = 3600 * 0.5

CONFIG_2020 = {
    "data_dir": "./data/model_data/2020_21/",
    "file_fixture": "fixtures.csv",
    "file_team": "teams.csv",
    "file_gw": "merged_gw.csv",
    "file_player": "players_raw.csv",
    "file_understat_team": "understat_team_data.pkl",
    "scoring_gw": "NA"
}


@cache.memoize(timeout=TIMEOUT_SHORT)
def query_next_gameweek():
    config = load_config()
    data_loader = DataLoader(config)
    next_gw = int(data_loader.get_next_gameweek_id())
    return next_gw


@cache.memoize(timeout=TIMEOUT)
def query_player_id_player_name_map():
    data_maker = ModelDataMaker(CONFIG_2020)
    player_id_player_name_map = data_maker.get_player_id_player_name_map()
    return player_id_player_name_map


@cache.memoize(timeout=TIMEOUT)
def query_league_standing():
    config = load_config()
    data_loader = DataLoader(config)
    df_league = data_loader.get_league_standings()
    return df_league
