from elmsley.runner.Runner import ECGClasssificationObject
import sys


def main(config, additional_argv):
    classifier_obj = ECGClasssificationObject(config_file_path=config.split("=")[1], argv=additional_argv)
    classifier_obj.run_experiment()


if __name__ == '__main__':
    main(config='config=config\\config.yml', additional_argv=None)
    # main(config=sys.argv[1], additional_argv=sys.argv[1:])