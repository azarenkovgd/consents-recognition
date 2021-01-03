from conrec import utils, conrec_class


def init():
    parameters = utils.load_json('parameters.json')  # загружает параметры
    conrec = conrec_class.ConRec(parameters)

    return conrec


def on_folder_files():
    conrec = init()
    conrec.on_multiple_files()


def find_values():
    conrec = init()
    conrec.find_values()


if __name__ == '__main__':
    find_values()
