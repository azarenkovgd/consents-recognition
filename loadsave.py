import pickle
import json
import os
from typing import Tuple

import cv2
import numpy as np
from pdf2image import convert_from_path


def prepare_orb_features_to_save(orb_features: tuple) -> tuple:
    """Подготавливает к сохранению orb_features.

    Необходимо из-за особенности opencv и pickle - cv2 keypoints не сохраняются через pickle напрямую.

    :param orb_features: orb фичи для сохранения.
    :return: orb фичи, подготовленные к сохранению с помощью pickle.
    """
    serialized_keypoints = []
    for point in orb_features[0]:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        serialized_keypoints.append(temp)

    return serialized_keypoints, orb_features[1]


def load_orb_features(path: str, max_features: int) -> tuple:
    """Загружает orb_features. Необходимо из-за особенности opencv и pickle - cv2 keypoints не сохраняются через
    pickle напрямую.

    :param path: откуда загрузить.
    :param max_features: количество фич для ORB классификатора.
    :return: orb фичи.
    """
    orb_features = load_pickle(path)[max_features]

    keypoints = []
    for point in orb_features[0]:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                            _octave=point[4], _class_id=point[5])
        keypoints.append(temp)

    return keypoints, orb_features[1]


def load_image(path: str) -> np.ndarray:
    """Загружает изображение, предотвращает продолжение в случае его отсутсвия на диске.

    :param path: путь к файлу.
    :return: изображение.
    :raises Exception: если файла не существует.
    """
    if not os.path.exists(path):
        raise Exception(f'Файла {path} не существует.')

    if os.path.splitext(path)[-1] == '.pdf':
        page = convert_from_path(path, 500)
        open_cv_image = np.array(page[0])
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image
    else:
        im = cv2.imread(path)
        return im


def save_pickle(obj, path: str):
    """Сохраняет файл в pickle формате.

    :param obj: объект для сохранения.
    :param path: путь для сохранения.
    """
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(path: str):
    """Загружает файл pickle формата.

    :param path: путь к объекту для загрузки.
    :return: загруженный объект.
    """
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)

    return obj


def load_fields(path: str) -> list:
    """Загружает поля из json файла.

    :param path: путь к файлу.
    :return: обработанные поля.
    """
    data = load_json(path)

    output = []
    for key in data:
        output.append([key] + data[key])

    return output


def load_template_values(number: int, max_features: int, data_folder: str = 'data') -> Tuple[np.ndarray, tuple, list]:
    """Загружает переменные связанные с исходным согласием

    :param number: номер согласия.
    :param max_features:
    :param data_folder: папка, в которой лежат данные связанные с согласиями.
    :return: данные об исходном согласии - файл, orb фичи, поля.
    """
    template = load_image(f'{data_folder}/sogl{number}_image.jpg')
    template_orb_features = load_orb_features(f'{data_folder}/sogl{number}_orb.pickle', max_features)
    fields = load_fields(f'{data_folder}/sogl{number}_fields.json')

    return template, template_orb_features, fields


def save_json(obj, path: str):
    """Сохраняет файл в формате json

    :param obj: объект для сохранения.
    :param path: путь для сохранения.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)


def load_json(path: str):
    """Загружает объект из файла формата json.

    :param path: путь к файлу.
    :return: загруженный объект.
    """
    with open(path, 'r', encoding='utf-8') as read_file:
        data = json.load(read_file)

    return data
