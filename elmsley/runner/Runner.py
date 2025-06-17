import os
import time
from loguru import logger
from alive_progress import alive_bar
from elmsley.config.Config import Config
from elmsley.internal.utils.json2dotnotation import banner
import importlib
import datetime


def camel_case(s):
    return ''.join(x for x in s.title() if not x.isspace()).replace('_', '')

class ECGClasssificationObject:


    def __init__(self, config_file_path='./config/config.yml', argv=None):
        """
        Initialize the ECGClasssification object.

        Args:
            config_file_path (str): The path to the configuration file (default is './config/config.yml').
                                    It can be either an absolute path or a path relative to the folder of the config file.
            argv (Optional[List[str]]): Additional arguments (default is None).

        Returns:
            None
        """

        if not os.path.exists('./local/logs/'):
            os.makedirs('./local/logs/')

        log_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # logging.basicConfig(
        #     level=logging.INFO,
        #     format="%(asctime)s [%(levelname)s] %(message)s",
        #     datefmt='%Y-%m-%d-%H:%M:%S',
        #     handlers=[
        #         logging.FileHandler(filename=f'./local/logs/{log_file}.log'),
        #         logging.StreamHandler()
        #     ]
        # )

        logger.add(f"./local/logs/{log_file}.log", level="INFO",
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")
        # logging.add(lambda msg: print(msg, end=""), level="INFO")  # Stream handler for console output # Deprecated

        logger.info('\n' + banner)
        logger.log("WELCOME",
                   '***  ELMSLEY')
        logger.log("WELCOME",
                   '*** Brought to you by: SisInfLab, Politecnico di Bari, Italy (https://sisinflab.poliba.it) ***\n')
        self._config = Config(config_file_path, argv)

    def run_experiment(self):
        """
        It executes all the experiment that have to be done
        """
        logger.info('Experiments is starting...')
        extractions_dict = self._config.get_parameters()
        for modality, modality_extractions in extractions_dict.items():
            for source, source_extractions in modality_extractions.items():
                self.do_extraction(modality, source)

        logger.success(f'Extraction is complete, it\'s coffee break! ☕️')


    def run_experiment(self):
        """
        It executes all the experiment that have to be done
        """
        logger.info('Experiments is starting...')
        parameters_dict = self._config.get_parameters()
        model_list = self._config._models

        for model in model_list:





        logger.success(f'Extraction is complete, it\'s coffee break! ☕️')


    def do_experiment(self, model):


