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


def preprocess_part_of_image_to_work(image: np.ndarray, x, y):
    return cv2.Canny(image[y[1]:y[0], x[1]:x[0]], 3, 100)


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
    # создание копии изображения для дальнейшего нанесения на него рамок полей
    debug_aligned_image: np.ndarray = aligned.copy()

    # изменение цвета изображения обратно из черно белого для удобства дальнейшего изучения
    debug_aligned_image = cv2.cvtColor(debug_aligned_image, cv2.COLOR_GRAY2BGR)

    num_of_filled_fields = 0

    # стандарт поля - (имя, y координаты рамки, x координаты рамки)
    for field in fields:
        y, x = field[1], field[2]

        # вырезание нужной части из присланной пользователем формы и шаблона для дальнейшей оценки
        img1 = preprocess_part_of_image_to_work(aligned, x, y)
        img2 = preprocess_part_of_image_to_work(template, x, y)

        mse_score_to_estimate_if_field_is_filled = mse(img1, img2)

        if mse_score_to_estimate_if_field_is_filled > threshold:
            num_of_filled_fields += 1

        if debug_mode == 1:
            # выбор цвета для рамки вокруг поля. зеленый если поле заполнено, красный если нет
            color = ((0, 47, 31) if mse_score_to_estimate_if_field_is_filled > threshold else (0, 0, 255))

            cv2.rectangle(debug_aligned_image, (x[0], y[0]), (x[1], y[1]), color, 3)
            cv2.putText(debug_aligned_image, str(int(mse_score_to_estimate_if_field_is_filled)),
                        (x[1], y[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    percent_filled = num_of_filled_fields / len(fields)
    return percent_filled, debug_aligned_image
