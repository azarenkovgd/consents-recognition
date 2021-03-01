import cv2
import numpy as np

from conrec import utils


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

    orb_features = utils.load_pickle(path)[max_features]

    keypoints = []
    for point in orb_features[0]:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                            _octave=point[4], _class_id=point[5])
        keypoints.append(temp)

    return keypoints, orb_features[1]


def create_orb_features(image: np.ndarray, max_features: int) -> tuple:
    """Создать ORB и применить к изображению.
    :param image: изображение для ORB.
    :param max_features: количество фич в ORB.
    :return: фичи, полученные с помощью ORB.
    """
    orb = cv2.ORB_create(max_features)
    image_orb_features = orb.detectAndCompute(image, None)
    return image_orb_features