import imutils

from conrec import utils, alignment, orbutils, recognition


class Image:
    def __init__(self, conrec, path_to_image):
        self.path_to_image = path_to_image

        self.image_to_align = None
        self.image_orb_features = None
        self.cached_matches = None
        self.aligned_image = None

        self.similarity_score = None
        self.percent_filled = None
        self.debug_aligned = None
        self.sha256_image = None
        self.is_correct = None

        self.conrec = conrec
        self.template = conrec.template
        self.template_image = self.template.template_image

    def load_image(self):
        """Загружает изображение и обрабатывает его"""
        image_to_align = utils.load_image(self.path_to_image)
        preprocessed_image = utils.preprocess_image(image_to_align)
        preprocessed_image = imutils.resize(preprocessed_image, self.template_image.shape[1])

        self.image_to_align = preprocessed_image
        self.sha256_image = utils.calc_sha256(self.path_to_image)

    def create_orb_for_image(self):
        """Генерирует орб фичи"""
        self.image_orb_features = orbutils.create_orb_features(self.image_to_align, self.conrec.max_features)

    def align_image(self):
        """Выравнивает изображение согласно шаблону"""
        pts1, pts2 = alignment.match_images(self.cached_matches, self.conrec.keep_percent,
                                            self.image_orb_features, self.template.template_orb_features)
        self.aligned_image = alignment.find_homography(self.image_to_align, self.template_image, pts1, pts2)

    def evaluate_aligned_image(self):
        """Оценивает выравненное изображение"""
        self.similarity_score = recognition.measure_similarity(self.aligned_image, self.template_image)
        values_to_unpack = recognition.check_right_filling(self.aligned_image, self.template_image,
                                                           self.template.fields, self.conrec.fields_threshold,
                                                           self.conrec.debug_mode)
        self.percent_filled, self.debug_aligned = values_to_unpack
