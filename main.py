import argparse
import os
import time

import cv2

import alignment
import recognition
import loadsave


def on_one_file(template, template_orb_features, fields: list, path: str, parameters: dict):
    start_time = time.time()

    image_to_align = loadsave.load_image(path)
    aligned = alignment.align_images(template, image_to_align, parameters['max_features'], parameters['keep_percent'],
                                     template_orb_features=template_orb_features)
    recognition_parameters = recognition.is_form_filled(aligned, template, fields,
                                                        fields_threshold=parameters['fields_threshold'],
                                                        similarity_threshold=parameters['similarity_threshold'],
                                                        percent_filled_threshold=parameters['percent_filled_threshold'],
                                                        debug_mode=parameters['debug_mode'])
    is_filled, debug_image, score, percent_filled = recognition_parameters

    end_time = round(time.time() - start_time, 5)

    output = [path, is_filled, score, percent_filled, end_time]
    return output


def on_multiple_files(path: str, number: int, parameters: dict):
    template, template_orb_features, fields = loadsave.load_template_values(number, parameters['max_features'],
                                                                            parameters['sogl_folder'])
    df = []
    headers = ['path', 'is_filled', 'score', 'percent_filled', 'end_time']

    for file_name in os.listdir(path):
        data = on_one_file(template, template_orb_features, fields, os.path.join(path, file_name), parameters)
        df.append(data)
        loadsave.save_csv(df, 'logs/logs.csv', headers)


def main():
    parser = argparse.ArgumentParser(description='Распознавание согласий.')
    parser.add_argument('path', type=str,
                        help='Путь к файлу согласия или к папке с файлами согласий.')
    parser.add_argument('number_of_consent', type=int,
                        help='Номер согласия, к которому относится присланный пользователем файл.')
    parser.add_argument('-p', '--parameters', type=str, default='parameters.json',
                        help='Путь к файлу с параметрами.')
    args = parser.parse_args()
    parameters = loadsave.load_json(args.parameters)  # загружает параметры

    if not os.path.isfile(args.path):
        on_multiple_files(args.path, args.number_of_consent, parameters)
    else:
        template, template_orb_features, fields = loadsave.load_template_values(args.number_of_consent,
                                                                                parameters['max_features'],
                                                                                parameters['sogl_folder'])
        on_one_file(template, template_orb_features, fields, args.path, parameters)


if __name__ == '__main__':
    main()
