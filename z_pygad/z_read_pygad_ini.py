import configparser


def read_config(config_path):

    config_dict = dict()

    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)

    for section in config_parser.sections():
        for argument, value in config_parser[section].items():
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value == 'False':
                        value = False
                    elif value == 'True':
                        value = True
                    elif value == 'None':
                        value = None

            config_dict[argument] = value

    return config_dict
