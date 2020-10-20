import argparse
import os

import cv2

import alignment
import recognition
import loadsave


def on_one_file(template, template_orb_features, fields: list, path: str, parameters: dict):
    """Запускает распознавание согласий на одном файле.

    :param parameters:
    :param template: образец согласия.
    :param template_orb_features: orb фичи, заренне расчитанные для согласия.
    :param fields: поля для сравнения.
    :param path: путь к файлу с загруженным согласием.
    """
    image_to_align = loadsave.load_image(path)
    aligned = alignment.align_images(template, image_to_align, parameters['max_features'], parameters['keep_percent'],
                                     template_orb_features=template_orb_features)
    recognition_parameters = recognition.is_form_filled(aligned, template, fields,
                                                        fields_threshold=parameters['fields_threshold'],
                                                        similarity_threshold=parameters['similarity_threshold'],
                                                        percent_filled_threshold=parameters['percent_filled_threshold'],
                                                        debug_mode=parameters['debug_mode'])
    is_filled, debug_image, score, percent_filled = recognition_parameters

    if is_filled:
        print(parameters['message if filled'])

    if parameters['debug_mode']:
        print(path, score, percent_filled)
        cv2.imwrite(os.path.join(parameters['debug_folder'], 'debug.jpg'), debug_image)
        cv2.imwrite(os.path.join(parameters['debug_folder'], 'debug_aligned.jpg'), aligned)


def on_multiple_files(path: str, number: int, parameters: dict):
    """Запуск распознавания согласий на всех файлах в директории path.

    :param parameters:
    :param path: путь к директории с файлами согласий.
    :param number: номер исходного согласия.
    """
    template, template_orb_features, fields = loadsave.load_template_values(number, parameters['max_features'],
                                                                            parameters['sogl_folder'])

    for file_name in os.listdir(path):
        on_one_file(template, template_orb_features, fields, os.path.join(path, file_name), parameters)


def main():
    parser = argparse.ArgumentParser(description='Распознавание согласий.')
    parser.add_argument('-f', action='store_true',
                        help='Если выбран, то скрипт будет запущен на всех файлах по указанному пути (path).'
                             'Если нет, то воспринимает путь (path) как местоположение файла и запустит скрипт на нем.')
    parser.add_argument('path', type=str,
                        help='Путь к файлу согласия или к папке с файлами согласий.')
    parser.add_argument('number', type=int,
                        help='Номер согласия, к которому относится присланный пользователем файл.')
    parser.add_argument('-p', '--parameters', type=str, default='parameters.json',
                        help='Путь к файлу с параметрами.')
    args = parser.parse_args()
    parameters = loadsave.load_json(args.parameters)  # загружает параметры

    if args.f:
        on_multiple_files(args.path, args.number, parameters)
    else:
        template, template_orb_features, fields = loadsave.load_template_values(args.number,
                                                                                parameters['max_features'],
                                                                                parameters['sogl_folder'])
        on_one_file(template, template_orb_features, fields, args.path, parameters)


if __name__ == '__main__':
    main()
