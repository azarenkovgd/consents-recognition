import os
import time

import imutils
import cv2

from conrec import utils, alignment, recognition, orbutils


class ConRec:
    def __init__(self, parameters):
        self.path_to_files = parameters['path_to_files']

        self.max_features = parameters['max_features']
        self.keep_percent = parameters['keep_percent']
        self.similarity_threshold = parameters['similarity_threshold']

        self.fields_threshold = parameters['fields_threshold']
        self.percent_filled_threshold = parameters['percent_filled_threshold']

        self.debug_mode = parameters['debug_mode']
        self.debug_folder = parameters['debug_folder']

        self.sogl_folder = parameters['sogl_folder']  # папка с файлами исходных согласий
        self.sogl_number = parameters['sogl_number']  # номер согласия

        template_values = utils.load_template_values(self.sogl_number, self.max_features, self.sogl_folder)
        self.template, self.template_orb_features, self.fields = template_values

        self.headers_for_logs_csv = ['path', 'score', 'percent_filled', 'is_correct', 'final_time']
        self.logs_dir = f'{parameters["logs_folder"]}/logs.csv'
        self.error_logs_dir = f'{parameters["logs_folder"]}/errors.csv'
        self.headers_for_error_csv = ['error_message', 'final_time']

        # при переборе наилучшей конфигурации можно будет сокращать время перебора, кешируя найденные значения
        self.cached_matches = None
        self.image_to_align = None
        self.image_orb_features = None
        self.aligned_image = None
        self.estimated_values = []
        self.debug_aligned = None

    def load_image(self, path_to_image):
        image_to_align = utils.load_image(path_to_image)
        preprocessed_image = utils.preprocess_image(image_to_align)
        preprocessed_image = imutils.resize(preprocessed_image, self.template.shape[1])

        self.image_to_align = preprocessed_image

    def create_orb_for_image(self):
        self.image_orb_features = orbutils.create_orb_features(self.image_to_align, self.max_features)

    def align_image(self):
        pts1, pts2 = alignment.match_images(self.cached_matches, self.image_orb_features, self.template_orb_features,
                                            self.keep_percent)
        self.aligned_image = alignment.find_homography(self.image_to_align, self.template, pts1, pts2)

    def evaluate_aligned_image(self):
        similarity_score = recognition.measure_similarity(self.aligned_image, self.template)
        percent_filled, debug_aligned = recognition.check_right_filling(self.aligned_image, self.template, self.fields,
                                                                        self.fields_threshold, self.debug_mode)

        if similarity_score > self.similarity_threshold and percent_filled > self.percent_filled_threshold:
            is_correct = True
        else:
            is_correct = False

        self.estimated_values = similarity_score, percent_filled, is_correct
        self.debug_aligned = debug_aligned

    def on_one_file(self):
        self.align_image()
        self.evaluate_aligned_image()

    def on_multiple_files(self):
        logs = []
        errors = []

        for file_name in os.listdir(self.path_to_files):
            time_start = time.time()
            data = [file_name]

            try:
                self.load_image(os.path.join(self.path_to_files, file_name))
                self.create_orb_for_image()
                self.on_one_file()
                data += list(self.estimated_values)

                final_time = round(time.time() - time_start, 3)
                data.append(final_time)

                logs.append(data)
                utils.save_csv(logs, self.logs_dir, self.headers_for_logs_csv)

                if self.debug_mode == 1:
                    cv2.imwrite(os.path.join(self.debug_folder, file_name), self.debug_aligned)

            except Exception as e:
                data.append(str(e))

                final_time = round(time.time() - time_start, 3)
                data.append(final_time)

                errors.append(data)
                utils.save_csv(errors, self.error_logs_dir, self.headers_for_error_csv)
