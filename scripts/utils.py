import os
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


if __name__ == "__main__":
    # this_path = Path(".")
    # print(this_path)
    # print(type(this_path))
    this_config = load_config()
    print(this_config)
