import json
import numpy as np
import cv2


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as read_file:
        data = json.load(read_file)

    return data


def create_orb_features(image: np.ndarray, max_features: int) -> tuple:
    """Создать ORB и применить к изображению.
    :param image: изображение для ORB.
    :param max_features: количество фич в ORB.
    :return: фичи, полученные с помощью ORB.
    """
    orb = cv2.ORB_create(max_features)
    image_orb_features = orb.detectAndCompute(image, None)
    return image_orb_features
