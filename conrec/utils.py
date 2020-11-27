import pickle
import json
import csv
import os
from typing import Tuple

import cv2
import numpy as np
from pdf2image import convert_from_path

from conrec import orbutils


def save_csv(data, path, headers):
    data = [headers] + data
    with open(path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)


def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise Exception(f'Файла {path} не существует.')

    if os.path.getsize(path) > 10000000:
        raise Exception(f'Файл {path} слишком большой')

    if os.path.getsize(path) < 40000:
        raise Exception(f'Файл {path} слишком маленький')

    if os.path.splitext(path)[-1] == '.pdf':
        page = convert_from_path(path, 500)
        open_cv_image = np.array(page[0])
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image
    else:
        im = cv2.imread(path)
        return im


def save_pickle(obj, path: str):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(path: str):
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)

    return obj


def load_fields(path: str) -> list:
    data = load_json(path)

    output = []
    for key in data:
        output.append([key] + data[key])

    return output


def save_json(obj, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as read_file:
        data = json.load(read_file)

    return data


def load_template_values(number: int, max_features: int, data_folder: str = 'data') -> Tuple[np.ndarray, tuple, list]:
    template = load_image(f'{data_folder}/sogl{number}_image.jpg')
    preprocessed_template = preprocess_image(template)

    template_orb_features = orbutils.create_orb_features(template, max_features)
    fields = load_fields(f'{data_folder}/sogl{number}_fields.json')

    return preprocessed_template, template_orb_features, fields


def preprocess_image(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image

