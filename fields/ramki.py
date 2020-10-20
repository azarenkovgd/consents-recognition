import cv2
import numpy as np


def load_image(folder, name):
    im = cv2.imread(f'{folder}/{name}')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def frame_img(img_f):
    hsv_frame = cv2.cvtColor(img_f, cv2.COLOR_BGR2HSV)

    # Диапазон цвета которым будем рисовать прямоугольники в любом редакторе
    # Цвет которого не может быть в документе (в нашем случае - синий)
    lower_range = np.array([110, 50, 50])
    upper_range = np.array([130, 255, 255])

    mask = cv2.inRange(hsv_frame, lower_range, upper_range)

    frames = []
    img_sk = mask
    thresh = cv2.Canny(img_sk, 10, 250)
    cnts_frame, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # перебираем все найденные контуры в цикле
    for cnt_frame in cnts_frame:
        rect = cv2.minAreaRect(cnt_frame)  # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        area = int(rect[1][0] * rect[1][1])  # вычисление площади
        if area > 250:  # функция подстраховки от случайных контуров (код будет работать без нее)
            cv2.drawContours(img_sk, [box], 0, (255, 0, 0), 2)
            frames.append(box[[1, 3]])

    array_frame = np.array(frames)
    return array_frame


if __name__ == "__main__":
    # OCR_LOCATIONS - готовый массив с координатами рамок проверки полей для основного проекта
    OCR_LOCATIONS = frame_img(cv2.imread('data/test.jpg'))  # Test - форма с синими прямоугольниками.

    template = load_image('data', '1.jpg')  # форма на которую будут накладываться вычисленные рамки

    for location in OCR_LOCATIONS:
        y, x = (location[0][1], location[1][1]), (location[0][0], location[1][0])
        cv2.rectangle(template, (x[0], y[0]), (x[1], y[1]), (0, 255, 0), 3)

    cv2.imwrite('new_fields', template)

