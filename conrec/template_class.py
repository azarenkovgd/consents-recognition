from conrec import utils, orbutils


class Template:
    def __init__(self, conrec):
        self.conrec = conrec

        self.template_image = None
        self.template_orb_features = None
        self.fields = None
        self.stop_sha256 = None

        self.load_template_values()

    def load_template_values(self):
        sogl_base_str = f'{self.conrec.sogl_folder}/sogl{self.conrec.sogl_number}'

        template_image = utils.load_image(sogl_base_str + '_image.jpg')
        self.template_image = utils.preprocess_image(template_image)

        self.fields = utils.load_fields(sogl_base_str + '_fields.json')
        self.stop_sha256 = utils.load_json('sogl_folder/sogl3_stop_sha256.json') #sogl_base_str + '_stop_sha256.json')

    def calc_orb(self):
        self.template_orb_features = orbutils.create_orb_features(self.template_image, self.conrec.max_features)
