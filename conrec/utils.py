import pickle
import json
import csv
import os
import hashlib

import cv2
import numpy as np
from pdf2image import convert_from_path


def calc_sha256(filename):
    sha256_hash = hashlib.sha256()

    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def write_row_csv(data, path, mode="a+", delimiter=',', newline=''):
    with open(path, mode=mode, newline=newline) as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=delimiter)
        csv_writer.writerow(data)


def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise Exception(f'Файла {path} не существует.')

    if os.path.getsize(path) > 10000000:
        raise Exception(f'Файл {path} слишком большой')

    if os.path.getsize(path) < 30000:
        raise Exception(f'Файл {path} слишком маленький')

    if os.path.splitext(path)[-1] == '.pdf':
        page = convert_from_path(path, dpi=500, single_file=True)
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
        json.dump(obj, f, ensure_ascii=False, indent=4)


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as read_file:
        data = json.load(read_file)

    return data


def preprocess_image(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image
