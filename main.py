from conrec import utils, conrec_class


def on_many_files():
    parameters = utils.load_json('parameters.json')  # загружает параметры

    conrec = conrec_class.ConRec(parameters)
    conrec.on_multiple_files()


def find_values():
    parameters = utils.load_json('parameters.json')  # загружает параметры

    conrec = conrec_class.ConRec(parameters)
    conrec.find_values()


if __name__ == '__main__':
    on_many_files()
