import cv2
import numpy as np


def create_orb_features(image: np.ndarray, max_features: int) -> tuple:
    """Создать ORB и применить к изображению.
    :param image: изображение для ORB.
    :param max_features: количество фич в ORB.
    :return: фичи, полученные с помощью ORB.
    """
    orb = cv2.ORB_create(max_features)
    image_orb_features = orb.detectAndCompute(image, None)
    return image_orb_features


def match_images(image_orb_features: tuple, template_orb_features: tuple, keep_percent: float) -> tuple:
    """Получить точки для find_homography.

    :param image_orb_features: orb фичи, полученнные из image_to_align с помощью ORB.
    :param template_orb_features: template фичи, полученные из template с помощью ORB.
    :param keep_percent: какой процент лучших фич оставить.
    :return: точки на image_orb_features и template_orb_features для find_homography.
    """
    (kps1, descs1) = image_orb_features
    (kps2, descs2) = template_orb_features

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    matches = matcher.match(descs1, descs2)
    matches = sorted(matches, key=lambda x: x.distance)

    number_of_matches_to_keep = int(len(matches) * keep_percent)
    matches = matches[:number_of_matches_to_keep]

    pts1 = np.zeros((len(matches), 2), dtype="float")
    pts2 = np.zeros((len(matches), 2), dtype="float")

    for (i, match_point) in enumerate(matches):
        pts1[i] = kps1[match_point.queryIdx].pt
        pts2[i] = kps2[match_point.trainIdx].pt

    return pts1, pts2


def find_homography(image_to_align: np.ndarray, template: np.ndarray, pts1: np.array, pts2: np.array):
    """Выровнять изображение.

    :param image_to_align: изображение для выравнивания.
    :param template: шаблон формы.
    :param pts1: точки на image_to_align.
    :param pts2: точки на template.
    :return: выравненное изображение.
    """

    (H, mask) = cv2.findHomography(pts1, pts2, method=cv2.RANSAC)

    (width, height) = template.shape[:2]
    aligned_image = cv2.warpPerspective(image_to_align, H, (height, width))

    return aligned_image
