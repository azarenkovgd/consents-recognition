from conrec import utils, conrec_class


def main():
    parameters = utils.load_json('parameters.json')  # загружает параметры

    conrec = conrec_class.ConRec(parameters)
    conrec.on_multiple_files()


if __name__ == '__main__':
    main()
