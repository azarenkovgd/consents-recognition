import cv2
import numpy as np
import os


def load_image(path: str) -> np.ndarray:
    """Загрузка ихображения

    :param path: путь к файлу с изображением
    :return: загруженное изображение
    """
    if type(path) != str:
        raise TypeError(f'Тип переменной path {type(path)} не является строкой')

    if not os.path.exists(path):
        raise FileNotFoundError(f'Файла {path} не существует.')

    image = cv2.imread(path)

    return image


def extract_contours(image: np.ndarray) -> np.ndarray:
    """Поиск контуров на изображени

    :param image: предварительно обработанное изображение с нанесенными рамками
    :return: контуры на изображении
    """
    # Диапазон цвета которого не может быть в документе (в нашем случае - синий)
    lower_range = np.array([110, 50, 50])
    upper_range = np.array([130, 255, 255])

    image_mask = cv2.inRange(image, lower_range, upper_range)

    thresh = cv2.Canny(image_mask, 10, 250)
    contours_of_frames, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours_of_frames


def extract_frames(contours_of_frames: np.ndarray) -> list:
    """Поиск рамок на изображении

    :param contours_of_frames: контуры на изображении
    :return: полученные координаты рамок
    """
    frames = []

    # перебираем все найденные контуры в цикле
    for contours_of_frame in contours_of_frames:
        rect = cv2.minAreaRect(contours_of_frame)  # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        area = int(rect[1][0] * rect[1][1])  # вычисление площади

        if area > 250:
            frames.append(box[[1, 3]])

    return np.array(frames).tolist()


def save_frames(path_to_image_with_fields: str,
                create_debug_form: bool = False, path_to_blank_image: str = None) -> (dict, np.ndarray):
    """По полученному изображению с нанесенными на него рамками (прямоугольники синего цвета по контурам мест, где
    пользователь будет вводить свои данные) вывести координаты полей, а также изображение с нанесенными на него заново
    рамками для проверки. Полученный в ходе работы данной программы массив значений можно использовать напрямую для
    подачи в текущую версию основного скрипта в качестве sogl{number}_fields.json.

    :param path_to_image_with_fields: путь к изображению с нанесенными рамками, которые выделяют нужные поля
    :param create_debug_form: если True, на загруженное пустое изображение наносятся рамки в соответствии с полученными
                            их координатами. это действие осуществляется для проверки, что все сработало нормально
    :param path_to_blank_image: пусть к исходному пустому изображению без нанесенных рамок
    :return: словарь с координатами полей, а также изображение с нанесенными заново рамками
    """
    image_with_frames = load_image(path_to_image_with_fields)
    image_with_frames = cv2.cvtColor(image_with_frames, cv2.COLOR_BGR2HSV)

    contours_of_frames = extract_contours(image_with_frames)
    frames = extract_frames(contours_of_frames)

    dict_with_values = {str(i): frame for i, frame in enumerate(frames)}

    if create_debug_form:
        template = load_image(path_to_blank_image)  # пустой бланк

        for location in frames:
            y = (location[0][1], location[1][1])
            x = (location[0][0], location[1][0])

            # нанесение рамок на форму по инструкции
            cv2.rectangle(template, (x[0], y[0]), (x[1], y[1]), (0, 255, 0), 3)

        return dict_with_values, template

    return dict_with_values, None
