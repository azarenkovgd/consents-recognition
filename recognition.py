import cv2
import numpy as np
import imutils
from typing import Tuple


def mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """Сравнивает два изображения с помощью mse. Сравнение банально и подходит только для самых простых примеров.

    :param image1: первое изображение для сравнения.
    :param image2: второе изображение для сравнения.
    :return: мера ошибки. Чем меньше, тем более похожи два изображения.
    """
    err = float(np.sum((image1.astype("float") - image2.astype("float")) ** 2))
    err /= float(image1.shape[0] * image1.shape[1])

    return err


def measure_similarity(aligned: np.ndarray, template: np.ndarray) -> float:
    """Сравнение изображений с помощью matchTemplate.

    В ходе экспериментов выяснилось, что работает лучше всего в задаче анализа того, насколько хорошо была выровнена
    форма. Используется изменение размеров и Canny для максимального устранения шумов и очистки изображения. Было также
    получено в ходе экспериментов. MSE, ssim и сравнение features из ORB не показало достаточных результатов.

    :param aligned: первое изображение для сравнения.
    :param template: второе изображение для сравнения.
    :return: значение от 0 до 1, чем больше, тем выше сходство.
    """
    result = cv2.matchTemplate(cv2.Canny(imutils.resize(template, width=600), 30, 100),
                               cv2.Canny(imutils.resize(aligned, width=600), 30, 100),
                               cv2.TM_CCORR_NORMED)
    (_, max_val, _, _) = cv2.minMaxLoc(result)

    return max_val


def check_right_filling(aligned: np.ndarray, template: np.ndarray, fields: list, threshold: float = 5000,
                        debug_mode: int = 0) -> Tuple[float, np.ndarray]:
    """Проверить, сколько в форме заполненных полей.

    :param aligned: выравненная форма, присланная пользователем.
    :param template: образец формы, незаполненный.
    :param fields: поля, для которых нужно произвести проверку.
    :param threshold: порог MSE от двух кусков изображения, после которого поле считается незаполненным.
    :param debug_mode: если равен 1, то сохранить форму с выделенными на ней зеленым и красным цветом заполненные и не
                    заполненные поля соответственно.
    :return: процент заполненных полей.
    """
    debug_aligned: np.ndarray = aligned.copy()
    debug_aligned = cv2.cvtColor(debug_aligned, cv2.COLOR_GRAY2BGR)

    num_of_filled = 0
    for field in fields:
        y, x = field[1], field[2]
        img1 = cv2.Canny(aligned[y[1]:y[0], x[1]:x[0]], 3, 100)
        img2 = cv2.Canny(template[y[1]:y[0], x[1]:x[0]], 3, 100)
        mse_score = mse(img1, img2)

        if mse_score > threshold:
            num_of_filled += 1

        if debug_mode > 0:
            color = ((0, 47, 31) if mse_score > threshold else (0, 0, 255))
            cv2.rectangle(debug_aligned, (x[0], y[0]), (x[1], y[1]), color, 3)
            cv2.putText(debug_aligned, str(int(mse_score)), (x[1], y[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    percent_filled = num_of_filled / len(fields)
    return percent_filled, debug_aligned


def is_form_filled(aligned: np.ndarray, template: np.ndarray, fields: list,
                   similarity_threshold: float = 0.3, fields_threshold: float = 5000,
                   percent_filled_threshold: float = 0.8, debug_mode: int = 0) -> Tuple[bool, np.ndarray, float, float]:
    """Считается ли форма заполненной или нет.

    :param aligned: выровненная форма.
    :param template: образец формы, незаполненный.
    :param fields: список данных о полях где пользователь должен был внести свои данные.
    :param similarity_threshold: порог схожести двух файлов, после которого форма считается правильно выровненной.
    :param fields_threshold: порог MSE от двух полей (заполненного и нет), после которого поле считается заполненным.
    :param percent_filled_threshold: порог, после которого форма считается заполненной.
    :param debug_mode: 0 - без дебаггинга, 1 - с.
    :return: заполнена ли форма или нет.
    """
    score = measure_similarity(aligned, template)
    percent_filled, debug_aligned = check_right_filling(aligned, template, fields, fields_threshold, debug_mode)
    is_filled = True

    if score < similarity_threshold or percent_filled < percent_filled_threshold:
        is_filled = False

    return is_filled, debug_aligned, score, percent_filled
