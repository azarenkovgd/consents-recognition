import time
import os

import cv2

from conrec import utils, image_class, template_class


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

        self.logs_folder = parameters["logs_folder"]
        self.headers_for_logs_csv = ['path', 'score', 'percent_filled', 'is_correct', 'sha256', 'final_time']
        self.logs_dir = f'{self.logs_folder}/logs.csv'
        self.error_logs_dir = f'{self.logs_folder}/errors.csv'
        self.headers_for_error_csv = ['path', 'error_message', 'final_time']

        self.logs = []
        self.errors = []

        self.image = None
        self.template = None

    def check_correctness_image(self):
        if (self.image.similarity_score > self.similarity_threshold and
            self.image.percent_filled > self.percent_filled_threshold and
                self.image.sha256_image not in self.template.stop_sha256):
            self.image.is_correct = True
        else:
            self.image.is_correct = False

    def on_one_image(self, path_to_image):
        self.image = image_class.Image(self, path_to_image)

        self.image.load_image()
        self.image.create_orb_for_image()
        self.image.align_image()
        self.image.evaluate_aligned_image()

        self.check_correctness_image()

    def on_one_template(self):
        self.template = template_class.Template(self)

        self.template.load_template_values()
        self.template.calc_orb()

    def one_iteration(self, time_start, file_name):
        abs_path_to_image = os.path.join(self.path_to_files, file_name)

        self.on_one_image(abs_path_to_image)

        data = [abs_path_to_image, self.image.similarity_score, self.image.percent_filled, self.image.is_correct,
                self.image.sha256_image]

        final_time = round(time.time() - time_start, 3)
        data.append(final_time)

        self.logs.append(data)
        utils.save_csv(self.logs, self.logs_dir, self.headers_for_logs_csv)

        if self.debug_mode == 1:
            cv2.imwrite(os.path.join(self.debug_folder, file_name), self.image.debug_aligned)

    def on_error(self, time_start, file_name, exception):
        data = [file_name, str(exception)]

        final_time = round(time.time() - time_start, 3)
        data.append(final_time)

        self.errors.append(data)
        utils.save_csv(self.errors, self.error_logs_dir, self.headers_for_error_csv)

    def on_multiple_files(self):
        self.on_one_template()

        for file_name in os.listdir(self.path_to_files):
            time_start = time.time()

            try:
                self.one_iteration(time_start, file_name)

            except Exception as e:
                self.on_error(time_start, file_name, e)
