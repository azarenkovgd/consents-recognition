import pickle
import json

import cv2
import numpy as np


def save_orb_features(orb_features: tuple, path: str):
    """Сохраняет orb_features. Необходимо из-за особенности opencv и pickle - cv2 keypoints не сохраняются через
    pickle напрямую.

    :param orb_features:
    :param path: путь для сохранения.
    """
    serialized_keypoints = []
    for point in orb_features[0]:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        serialized_keypoints.append(temp)

    save_pickle((serialized_keypoints, orb_features[1]), path)


def load_orb_features(path: str) -> tuple:
    """Загружает orb_features. Необходимо из-за особенности opencv и pickle - cv2 keypoints не сохраняются через
    pickle напрямую.

    :param path: откуда загрузить.
    :return:
    """
    orb_features = load_pickle(path)

    keypoints = []
    for point in orb_features[0]:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                            _octave=point[4], _class_id=point[5])
        keypoints.append(temp)

    return keypoints, orb_features[1]


def load_image(folder: str, name: str) -> np.ndarray:
    im = cv2.imread(f'{folder}/{name}')
    return im


def save_pickle(obj, path: str):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(path: str):
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)

    return obj


def load_fields(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as read_file:
        data = json.load(read_file)

    output = []
    for key in data:
        output.append([key] + data[key])

    return output

