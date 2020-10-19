import argparse
import os

import alignment
import recognition
import loadsave


PARAMETERS = {
    'message if filled': 'kruzhok',
    'max_features': 3000,
    'keep_percent': 0.4,
    'similarity_threshold': 0.3,
    'fields_threshold': 4000,
    'percent_filled_threshold': 0.8,
    'debug_mode': 0,
}


def on_one_file(template, template_orb_features, fields, path):
    """Запускает распознавание согласий на одном файле.

    :param template: образец согласия.
    :param template_orb_features: orb фичи, заренне расчитанные для согласия.
    :param fields: поля для сравнения.
    :param path: путь к файлу с загруженным согласием.
    """
    image_to_align = loadsave.load_image(path)
    aligned = alignment.align_images(template, image_to_align, PARAMETERS['max_features'], PARAMETERS['keep_percent'],
                                     template_orb_features=template_orb_features)
    is_filled, _, _, _ = recognition.is_form_filled(aligned, template, fields,
                                                    fields_threshold=PARAMETERS['fields_threshold'],
                                                    similarity_threshold=PARAMETERS['similarity_threshold'],
                                                    percent_filled_threshold=PARAMETERS['percent_filled_threshold'],
                                                    debug_mode=PARAMETERS['debug_mode'])

    if is_filled:
        print(PARAMETERS['message if filled'])


def on_multiple_files(path, number):
    """Запуск распознавания согласий на всех файлах в директории path.

    :param path: путь к директории с файлами согласий.
    :param number: номер исходного согласия.
    """
    template, template_orb_features, fields = loadsave.load_template_values(number, PARAMETERS['max_features'])

    for file_name in os.listdir(path):
        on_one_file(template, template_orb_features, fields, os.path.join(path, file_name))


def main():
    parser = argparse.ArgumentParser(description='Распознавание согласий.')
    parser.add_argument('-f', action='store_true',
                        help='Если выбран, то скрипт будет запущен на всех файлах по указанному пути (path).'
                             'Если нет, то воспринимает путь (path) как местоположение файла и запустит скрипт на нем.')
    parser.add_argument('path', type=str,
                        help='Путь к файлу согласия или к папке с файлами согласий.')
    parser.add_argument('number', type=int,
                        help='Номер согласия, к которому относится присланный пользователем файл.')
    args = parser.parse_args()

    if args.f:
        on_multiple_files(args.path, args.number)
    else:
        template, template_orb_features, fields = loadsave.load_template_values(args.number, PARAMETERS['max_features'])
        on_one_file(template, template_orb_features, fields, args.path)


if __name__ == '__main__':
    main()
