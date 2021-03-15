import json
import numpy as np
import cv2


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as read_file:
        data = json.load(read_file)

    return data
