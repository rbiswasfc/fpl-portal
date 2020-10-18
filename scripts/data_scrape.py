import os
import sys
import json
import requests

import pandas as pd
from datetime import datetime

sys.path.insert(0, './')
from scripts.utils import load_config, check_create_dir


def fetch_data(url):
    """
    get json data from API
    :param url: web url
    :type url: str
    :return: json data
    :rtype: dict
    """
    response = requests.get(url)
    rcode = response.status_code
    if rcode != 200:
        raise Exception("Response was code {} for this url: {}".format(rcode, url))
    response_str = response.text
    res_dict = json.loads(response_str)
    return res_dict


class DataScraper(object):
    """
    Provides interface to fetch fantasy premier league data
    """

    def __init__(self, config):
        """
        initialize data scraper
        :param config: config file specifying filepaths
        :type config: dict
        """
        self.name = "FPL-Scraper"
        self.config = config
        self.data = None
        self.data_dir = os.path.join(config["source_dir"], config["season"])
        check_create_dir(self.data_dir)
        # set fpl urls
        self.fpl_url = "https://fantasy.premierleague.com/api/"
        self.login_url = "https://users.premierleague.com/accounts/login/"
        self.manager_url = "https://fantasy.premierleague.com/api/entry/"
        self.classic_league_suburl = "leagues-classic/"
        self.team_entry_suburl = "entry/"
        self.bootstrap_suburl = "bootstrap-static/"
        self.player_suburl = "element-summary/"
        self.fixtures_suburl = "fixtures/"

        self.league_standing_url = self.fpl_url + self.classic_league_suburl

        try:
            self.username = os.environ["fpl_email"]
            self.password = os.environ["fpl_pwd"]
        except:
            print("Error: Set FPL Email and Password in your OS environment")

        payload = {
            'login': self.username,
            'password': self.password,
            'redirect_uri': "https://fantasy.premierleague.com/",
            'app': 'plfpl-web'
        }

        self.session = requests.session()
        self.session.post(self.login_url, data=payload)

    def get_bootstrap_data(self):
        """
        Retrieve the fpl player data from static url
        """
        url = self.fpl_url + self.bootstrap_suburl
        data = fetch_data(url)
        return data

    def get_fixtures_data(self):
        url = self.fpl_url + self.fixtures_suburl
        data = fetch_data(url)
        return data

    def get_gameweek_data(self):
        bootstrap_data = self.get_bootstrap_data()
        gameweek_data = bootstrap_data["events"] 
        return gameweek_data

    def get_all_players_data(self):
        """
        Fetch player specific data from the FPL REST API endpoint
        :return:
        :rtype:
        """

        bootstrap_data = self.get_bootstrap_data()
        players = bootstrap_data['elements']
        base_url = self.fpl_url + self.player_suburl

        for player in players:
            player_id = player['id']
            player_url = base_url + "/{}/".format(player_id)
            data = fetch_data(player_url)
            player['history'] = data['history']
        return players

    def get_player_data(self, player_id):
        url = self.fpl_url + self.player_suburl + "/{}/".format(player_id)
        data = fetch_data(url)
        return data
        

    def get_team_data(self):
        """
        Fetch teams data
        :return:
        :rtype:
        """
        data = self.get_bootstrap_data()
        teams_data = data['teams']
        return teams_data

    def save_json_data(self, filename, data):
        """
        save data extracted from the FPL api
        :param filename:
        :type filename:
        :param data:
        :type data:
        :return:
        :rtype:
        """
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def get_fpl_manager_entry_ids(self, league_id="1457340"):
        entries = []
        entry_names = []
        player_names = []
        points = []
        ranks = []
        ls_page = 1
        while (True):
            league_url = self.league_standing_url + str(
                league_id) + "/standings/" + "?page_new_entries=1&page_standings=" + str(ls_page) + "&phase=1"
            # print(league_url)
            response = self.session.get(league_url)
            json_response = response.json()
            managers = json_response["standings"]["results"]
            if not managers:
                print("Total managers: {}".format(len(entries)))
                break
            for manager in managers:
                entries.append(manager["entry"])
                entry_names.append(manager["entry_name"])
                player_names.append(manager["player_name"])
                points.append(manager["total"])
                ranks.append(manager["rank"])
            ls_page += 1
        df_manager = pd.DataFrame()
        df_manager["entry_id"] = entries
        df_manager["entry_name"] = entry_names
        df_manager["manager_name"] = player_names
        df_manager["score"] = points
        df_manager["rank"] = ranks
        return df_manager

    def get_entry_data(self, entry_id):
        """
        Fet team picks for a particular FPL manager
        :param entry_id:
        :type entry_id:
        :return:
        :rtype:
        """

        url = self.manager_url + str(entry_id) + "/history/"
        data = fetch_data(url)
        return data

    def get_entry_personal_data(self, entry_id):
        """
        get personal data of FPL manager
        :param entry_id:
        :type entry_id:
        :return:
        :rtype:
        """
        url = self.manager_url + str(entry_id) + "/"
        data = fetch_data(url)
        return data

    def get_entry_gws_data(self, entry_id, num_gws):
        """
        get gameweek picks of fantasy managers
        :param entry_id:
        :type entry_id:
        :param num_gws: how many gw data to extract
        :type num_gws: int
        :return:
        :rtype:
        """
        gw_data = []
        for i in range(1, num_gws + 1):
            url = self.manager_url + str(entry_id) + "/event/" + str(i) + "/picks/"
            data = fetch_data(url)
            gw_data += [data]
        return gw_data

    def get_entry_transfers_data(self, entry_id):
        """
        get transfer data of fpl managers
        :param entry_id:
        :type entry_id:
        :return:
        :rtype:
        """
        url = self.manager_url + str(entry_id) + "/transfers/"
        data = fetch_data(url)
        return data

    def get_recent_gameweek_id(self):
        gws = self.get_gameweek_data()
        now = datetime.utcnow()
        for gw in gws:
            deadline = datetime.strptime(gw['deadline_time'], '%Y-%m-%dT%H:%M:%SZ')
            if deadline > now:
                return gw['id'] - 1

    def execute_all(self):
        pass


if __name__ == "__main__":
    this_config = {"season": "2020_21", "source_dir": "./data/raw/"}
    data_scraper = DataScraper(this_config)
    
    # gw_data = data_scraper.get_gameweek_data()
    # print(gw_data[2])
    # print(len(gw_data))

    # this_player_id = 4
    # this_player_data = data_scraper.get_player_data(this_player_id)
    # print(this_player_data.keys())
    # print(this_player_data["history"][0].keys())

    cur_gw = data_scraper.get_recent_gameweek_id()
    print(cur_gw)

    # df = data_scraper.get_fpl_manager_entry_ids()
    # print(df)
    # this_entry_id = '2235933'
    # gws = 4

    # fixtures_data = data_scraper.get_fixtures_data()
    # print(fixtures_data[5])
    # print(len(fixtures_data))

    # personal_data = data_scraper.get_entry_personal_data(this_entry_id)
    # print(personal_data)

    # entry_data = data_scraper.get_entry_data(this_entry_id)
    # print(entry_data)

    # entry_gw_data = data_scraper.get_entry_gws_data(this_entry_id, gws)
    # print(entry_gw_data)

    # transfer_data = data_scraper.get_entry_transfers_data(this_entry_id)
    # print(transfer_data)