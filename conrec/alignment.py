import cv2
import numpy as np


def match_images(matches, image_orb_features: tuple, template_orb_features: tuple, keep_percent: float) -> tuple:
    """Получить точки для find_homography.

    :param matches: предсозданные параметры
    :param image_orb_features: orb фичи, полученнные из image_to_align с помощью ORB.
    :param template_orb_features: template фичи, полученные из template с помощью ORB.
    :param keep_percent: какой процент лучших фич оставить.
    :return: точки на image_orb_features и template_orb_features для find_homography.
    """
    (kps1, descs1) = image_orb_features
    (kps2, descs2) = template_orb_features

    if not matches:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
        matches = matcher.match(descs1, descs2)
        matches = sorted(matches, key=lambda x: x.distance)

    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]

    pts1 = np.zeros((len(matches), 2), dtype="float")
    pts2 = np.zeros((len(matches), 2), dtype="float")

    for (i, m) in enumerate(matches):
        pts1[i] = kps1[m.queryIdx].pt
        pts2[i] = kps2[m.trainIdx].pt

    return pts1, pts2


def find_homography(image_to_align: np.ndarray, template: np.ndarray, pts1: np.array, pts2: np.array):
    """Выровнять изображение.

    :param image_to_align: изображение для выравнивания.
    :param template: изображение ориентир.
    :param pts1: точки на image_to_align.
    :param pts2: точки на template.
    :return: выравненное изображение.
    """

    (H, mask) = cv2.findHomography(pts1, pts2, method=cv2.RANSAC)

    (h, w) = template.shape[:2]
    aligned_image = cv2.warpPerspective(image_to_align, H, (w, h))

    return aligned_image

