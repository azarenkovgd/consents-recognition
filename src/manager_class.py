import image_class
import template_class


class Manager:
    def __init__(self, parameters):
        self.max_number_of_features_to_create = parameters['max_number_of_features_to_create']
        self.percent_of_features_to_keep = parameters['percent_of_features_to_keep']
        self.threshold_of_filling_for_fields = parameters['threshold_of_filling_for_fields']

        self.max_possible_correctness_of_alignment_left = parameters['max_possible_correctness_of_alignment_left']
        self.max_number_of_filled_fields_left = parameters['max_number_of_filled_fields_left']
        self.max_possible_correctness_of_alignment_right = parameters['max_possible_correctness_of_alignment_right']
        self.max_number_of_filled_fields_right = parameters['max_number_of_filled_fields_right']

        self.debug_mode = parameters['debug_mode']  # сохранять ли развернутые изображения на диск для оценки
        self.debug_folder = parameters['debug_folder']  # куда сохранять их

        # папка с файлами шаблонов и данными для них
        self.folder_with_template_data = parameters['folder_with_template_data']
        self.template_number = parameters['template_number']  # номер согласия

        self.image = None
        self.template = None

    def init_image(self, path_to_image: str):
        """Инициализирует изображение для дальнейшей работы"""
        self.image = image_class.Image(self)
        self.image.load_image(path_to_image)
        self.image.calculate_sha256()

    def init_template(self):
        """Подгружает шаблон и связанные с ним данные. Обычно выполняется один раз, а потом много раз используется"""
        self.template = template_class.Template(self)

        self.template.load_template_values()
        self.template.calc_orb()

    def check_if_image_hash_was_detected_before(self):
        if self.image.sha256_of_image in self.template.sha256_stop_list:
            return self.template.sha256_stop_list[self.image.sha256_of_image]

        return 0

    def main_calculations_for_sent_image(self):
        """Запускает программу на одном изображении, проводя все необходимые операции для получения оценки изображения
        на корректность"""
        self.image.preprocess_image()
        self.image.create_orb_for_image()
        self.image.align_image()
        self.image.evaluate_aligned_image()

    def classify_image_if_checks_passed(self):
        """Определение типа присланного изображения по предварительно вычисленным параметрам"""
        if (self.image.similarity_score < self.max_possible_correctness_of_alignment_left and
                self.image.percent_filled < self.max_number_of_filled_fields_left):
            return 1

        if (self.image.similarity_score > self.max_possible_correctness_of_alignment_right and
                self.image.percent_filled > self.max_number_of_filled_fields_right):
            return 3

        return 2

    def get_type_of_image(self, path_to_image: str) -> (int, str):
        """Определение типа присланного изображения по предварительно вычисленным параметрам

        :param path_to_image: абсолютный или относительный путь к изображению.
        :return: первое значение - тип изображения, второе - человеко читаемое описание и дополнительные данные.
        """
        try:
            self.init_image(path_to_image)

            number_of_matching_hashes = self.check_if_image_hash_was_detected_before()
            if number_of_matching_hashes > 0:
                return 4, f"Изображение уже было встречено {number_of_matching_hashes} раз"

            self.main_calculations_for_sent_image()
            type_of_sent_image = self.classify_image_if_checks_passed()
            self.image.save_debug_image_if_needed()

            return type_of_sent_image, f"Изображение было отнесено к типу {type_of_sent_image}." \
                                       f"Система дает оценку {self.image.similarity_score} " \
                                       f"качеству приведения его к нормальному виду и оценивает" \
                                       f"Количество заполненных полей в {self.image.percent_filled}."

        except Exception as e:
            return 5, f"Выполнение было прервано с ошибкой {str(e)}"
