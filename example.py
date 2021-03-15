from conrec import utils, manager_class


def main():
    parameters = utils.load_json('parameters.json')
    manager = manager_class.Manager(parameters)
    manager.init_template()
    type_of_image, description = manager.get_type_of_image('path/to/image.jpg')


if __name__ == '__main__':
    main()
