import os
import hashlib

import cv2
import numpy as np
from pdf2image import convert_from_path


def calc_sha256(path: str) -> str:
    sha256_hash = hashlib.sha256()

    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def load_pdf_and_convert_to_opencv_image(path: str) -> np.ndarray:
    # single_file - параметр, который говорит утилите, чтобы она использовала только первую страницу pdf файла
    page = convert_from_path(path, dpi=500, single_file=True)
    open_cv_image = np.array(page[0])
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    return open_cv_image


def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise Exception(f'Файла {path} не существует.')

    if os.path.getsize(path) > 10000000:
        raise Exception(f'Файл {path} слишком большой')

    if os.path.getsize(path) < 10000:
        raise Exception(f'Файл {path} слишком маленький')

    if os.path.splitext(path)[-1] == '.pdf':
        page = convert_from_path(path, dpi=500, single_file=True)
        open_cv_image = np.array(page[0])
        open_cv_image = open_cv_image[:, :, ::-1].copy()
    else:
        open_cv_image = cv2.imread(path)

    return open_cv_image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
