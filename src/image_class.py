import os

import cv2

import imutils
import alignment
import recognition
import image_preprocessing
import manager_class


class Image:
    def __init__(self, manager: 'manager_class.Manager'):
        self.path_to_image = None

        self.image_to_align = None
        self.image_orb_features = None
        self.aligned_image = None

        self.similarity_score = None
        self.percent_filled = None
        self.debug_aligned = None
        self.sha256_of_image = None

        self.manager = manager
        self.template_class = manager.template
        self.template_image = self.template_class.template_image

    def load_image(self, path_to_image):
        """Загружает изображение и обрабатывает его"""
        self.path_to_image = path_to_image
        self.image_to_align = image_preprocessing.load_image(self.path_to_image)

    def calculate_sha256(self):
        self.sha256_of_image = image_preprocessing.calc_sha256(self.path_to_image)

    def preprocess_image(self):
        """Обрабатывает присланное изображение"""
        preprocessed_image = image_preprocessing.preprocess_image(self.image_to_align)
        preprocessed_image = imutils.resize(preprocessed_image, self.template_image.shape[1])

        self.image_to_align = preprocessed_image

    def create_orb_for_image(self):
        """Генерирует орб фичи"""
        self.image_orb_features = alignment.create_orb_features(self.image_to_align,
                                                                self.manager.max_number_of_features_to_create)

    def align_image(self):
        """Выравнивает изображение согласно шаблону"""
        pts1, pts2 = alignment.match_images(self.image_orb_features, self.template_class.template_orb_features,
                                            self.manager.percent_of_features_to_keep)
        cv2.imwrite(f'../debug/t20.jpg', self.image_to_align)
        self.aligned_image = alignment.find_homography(self.image_to_align, self.template_image, pts1, pts2)
        cv2.imwrite(f'../debug/t21.jpg', self.aligned_image)

    def evaluate_aligned_image(self):
        """Оценивает выравненное изображение"""
        self.similarity_score = recognition.measure_similarity(self.aligned_image, self.template_image)
        values_to_unpack = recognition.check_right_filling(self.aligned_image, self.template_image,
                                                           self.template_class.fields,
                                                           self.manager.threshold_of_filling_for_fields,
                                                           self.manager.debug_mode)
        self.percent_filled, self.debug_aligned = values_to_unpack

    def save_debug_image_if_needed(self):
        # при работе программы осуществляется разворот присланного изображения в соответствии с шаблоном.
        # если debug_mode равен 1, осуществляется сохранение изображения для дальнейшего анализа.
        if self.manager.debug_mode == 1:
            debug_file_name = self.path_to_image.split('/')[-1]
            debug_file_name = debug_file_name.split('.')[0] + '.jpg'

            path = os.path.join(self.manager.debug_folder, debug_file_name)
            cv2.imwrite(path, self.debug_aligned)

            print('saved debug_image')
