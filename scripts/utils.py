import os
import time
import json
from pathlib import Path


def check_create_dir(dir_path):
    """
    check if folder exists at a specific path, if not create the directory

    :param dir_path: path to the directory to be created
    :type dir_path: str
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def load_config():
    with open("./assets/config.json", 'r') as f:
        config = json.load(f)
    return config


def check_cache_validity(file_path, valid_days=0.15):
    """
    Check cache validity for particular file
    :param file_path: path to the file
    :type file_path: str
    :param valid_days: cache validity threshold
    :type valid_days: float
    """
    if os.path.isfile(file_path):
        mod_time = os.path.getmtime(file_path)
        if (time.time() - mod_time) / 3600 < 24 * valid_days:
            return True
    return False


if __name__ == "__main__":
    # this_path = Path(".")
    # print(this_path)
    # print(type(this_path))
    this_config = load_config()
    print(this_config)
