import cv2
import numpy as np
import imutils


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


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Препроцессинг изображения.

    :param image: изображение для обработки.
    :return: обработанное изображение.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def align_images(template, image_to_align: np.ndarray, max_features: int, keep_percent: float,
                 image_orb_features: tuple = (None, None), template_orb_features: tuple = (None, None)) -> np.ndarray:
    """Выровнять image_to_align, используя template как ориентир.

    :param template: изображение ориентир.
    :param image_to_align: изображение для выравнивания.
    :param max_features: сколько фич использовать в cv2 orb.
    :param keep_percent: какой процент лучших фич оставить.
    :param image_orb_features: если image_orb_features были уже вычислены изначально, указать здесь.
    :param template_orb_features: если template_orb_features были уже вычислены изначально, указать здесь.
    :return: выровненное изображение.
    """
    template = preprocess_image(template)
    # изменение размера изображения для устранения слишком высокого качества присланной формы.
    image_to_align = imutils.resize(preprocess_image(image_to_align), template.shape[1])

    if not all(template_orb_features):  # Если template_orb_features равен (None, None)
        template_orb_features = create_orb_features(template, max_features)

    if not all(image_orb_features):  # Если image_orb_features равен (None, None)
        image_orb_features = create_orb_features(image_to_align, max_features)

    pts1, pts2 = match_images(image_orb_features, template_orb_features, keep_percent)
    aligned_image = find_homography(image_to_align, template, pts1, pts2)

    return aligned_image
