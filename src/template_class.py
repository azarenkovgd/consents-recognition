import image_preprocessing
import utils

import manager_class
import alignment


def load_fields(path: str) -> list:
    """Загрузка информации о полях у шаблона"""
    data = utils.load_json(path)

    output = []
    for key in data:
        output.append([key] + data[key])

    return output


class Template:
    def __init__(self, manager: 'manager_class.Manager'):
        self.manager = manager

        self.template_image = None
        self.fields = None
        self.sha256_stop_list = None

        self.template_orb_features = None

        self.template_path_prefix = f'{self.manager.folder_with_template_data}/sogl{self.manager.template_number}'

    def load_template_values(self):
        """Загружает шаблон, поля, хэш"""

        template_image = image_preprocessing.load_image(self.template_path_prefix + '_image.jpg')
        self.template_image = image_preprocessing.preprocess_image(template_image)

        self.fields = load_fields(self.template_path_prefix + '_fields.json')
        self.sha256_stop_list = utils.load_json(self.template_path_prefix + '_stop_sha256.json')

    def calc_orb(self):
        """Вычисляет орб фичи"""
        self.template_orb_features = alignment.create_orb_features(self.template_image,
                                                                   self.manager.max_number_of_features_to_create)

    def update_sha256_stop_list(self):
        """Обновление списка хэшей для определения, была ли форма уже отправлена или нет"""
        self.sha256_stop_list = utils.load_json(self.template_path_prefix + '_stop_sha256.json')
