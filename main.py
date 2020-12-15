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


def on_selected_filed():
    conrec = init()

    conrec.on_multiple_files(on_selected_files=True)


if __name__ == '__main__':
    on_selected_filed()
