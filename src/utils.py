import json


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as read_file:
        data = json.load(read_file)

    return data
