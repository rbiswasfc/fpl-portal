import os
import sys
import pickle

sys.path.insert(0, './')
from scripts.data_scrape import get_understat_epl_season_data
from scripts.data_processor import DataProcessor


class DataIngestor(object):
    def __init__(self, config):
        self.config = config
        self.ingest_dir = config["ingest_dir"]
        self.player_filepath = os.path.join(config["ingest_dir"], config["player_ingest_filename"])
        self.gw_filepath = os.path.join(config["ingest_dir"], config["gw_ingest_filename"])
        self.team_filepath = os.path.join(config["ingest_dir"], config["team_ingest_filename"])
        self.understat_filepath = os.path.join(config["ingest_dir"], config["understat_ingest_filename"])
        self.fixture_filepath = os.path.join(config["ingest_dir"], config["fixture_ingest_filename"])

    def ingest_player_data(self):
        data_processor = DataProcessor(self.config)
        df_players_raw = data_processor.save_players_data()
        df_players_raw.to_csv(self.player_filepath, index=False)

    def ingest_team_data(self):
        data_processor = DataProcessor(self.config)
        df_teams = data_processor.save_teams_data()
        df_teams.to_csv(self.team_filepath, index=False)

    def ingest_fixture_data(self):
        data_processor = DataProcessor(self.config)
        df_fixture = data_processor.save_fixtures_data()
        df_fixture.to_csv(self.fixture_filepath, index=False)

    def ingest_understat_data(self):
        understat_season = self.config["season"].split('_')[0]
        print(understat_season)
        team_data, _ = get_understat_epl_season_data(understat_season)
        with open(self.understat_filepath, 'wb') as f:
            pickle.dump(team_data, f)

    def ingest_gw_data(self):
        data_processor = DataProcessor(self.config)
        df_gw_merged = data_processor.save_gameweek_data()
        df_gw_merged.to_csv(self.gw_filepath, index=False)


if __name__ == "__main__":
    config_2020 = {"season": "2020_21",
                   "source_dir": "./data",
                   "ingest_dir": "./data/model_data/2020_21/",
                   "player_ingest_filename": "players_raw.csv",
                   "team_ingest_filename": "teams.csv",
                   "gw_ingest_filename": "merged_gw.csv",
                   "understat_ingest_filename": "understat_team_data.pkl",
                   "fixture_ingest_filename": "fixtures.csv"
                   }
    data_ingestor = DataIngestor(config_2020)
    data_ingestor.ingest_player_data()
    data_ingestor.ingest_team_data()
    data_ingestor.ingest_fixture_data()
    data_ingestor.ingest_understat_data()
    data_ingestor.ingest_gw_data()
