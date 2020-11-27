import os
from conrec import utils, conrec_body, orbutils


def main():
    parameters = utils.load_json('parameters.json')  # загружает параметры

    conrec = conrec_body.ConRec(parameters)

    if not os.path.isfile(conrec.path_to_files):
        conrec.on_multiple_files()
    else:
        conrec.load_image(conrec.path_to_files)
        conrec.create_orb_for_image()


if __name__ == '__main__':
    main()
