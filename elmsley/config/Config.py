import os
from loguru import logger
import json
from elmsley.internal.utils.YamlFileManagerParam import YamlFileManagerParam
from elmsley.internal.utils.YamlFileManagerModels import YamlFileManagerModels
from elmsley.internal.utils.json2dotnotation import parse_and_print
import warnings
import copy


def _clean_preprocessing_flag_of_models(model, type_of_extraction):
    """
    Clean preprocessing flags of models by renaming them under the same name for future data management.

    Args:
        model: The model object.
        type_of_extraction: The type of data extraction: Textual, Visual, and Audio

    Returns:
        dict: The model object with the renamed preprocessing flag.
    """
    data_flag = ''

    if type_of_extraction == 'textual':
        data_flag = model.pop('clear_text') if 'clear_text' in model.keys() else False
    elif type_of_extraction == 'visual':
        if 'reshape' in model.keys():
            data_flag = model.pop('reshape')
            if 'transformers' in model['backend']:
                logger.warning(f"Custom reshape may be overridden by predefined HuggingFace transformers preprocessing module's configurations")

    elif type_of_extraction == 'audio':
        # Right now there is no preprocessing flag but one is needed for code clearance
        data_flag = None

    model.update({'preprocessing_flag': data_flag})
    return model


def _clean_unique_flags_of_models(model, type_of_extraction):
    if type_of_extraction == 'textual':
        # to maintain the runner agnostic, when it gives the model name to the extractor, it also need to give it the
        # task that the model have to do.
        # so in textual...
        print('nah, after')


class Config:
    """
    Manage the configuration within the config YAML file.

    These configurations are needed to define what extracions to perform.

    """
    def __init__(self, param_config_file_path, model_config_file_path):
        """
        Initialize ConfigurationManager with the specified configuration file path and command-line arguments.

        Args:
            config_file_path (str): Path to the config YAML file.
            argv (list): Runner's arguments.

        Returns:
            None
        """
        # both absolute and relative path are fine
        self._param_yaml_manager = YamlFileManagerParam(param_config_file_path)
        self._param_data_dict = self._param_yaml_manager.get_default_dict()
        custom_param_data_dict = self._param_yaml_manager.get_raw_dict()
        self._param_data_dict.update(custom_param_data_dict)

        self._model_yaml_manager = YamlFileManagerModels(model_config_file_path)
        self._models = self._model_yaml_manager.load_models_from_yaml()


        if param_config_file_path != './config/config.yml':
            logger.warning(f'Custom configuration file {param_config_file_path} provided. Will override the default one')
        else:
            logger.info('No custom configuration file provided. Will use the default one')

        if model_config_file_path != './config/modelMap_old.yml':
            logger.warning(f'Custom configuration file {model_config_file_path} provided. Will override the default one')
        else:
            logger.info('No custom configuration file provided. Will use the default one')



        self._param_data_dict = self.__clean_dict(self._param_data_dict)

        logger.info(f'Loaded parameters configuration:\n\n{parse_and_print(self._param_data_dict)}\n')




    def __clean_dict(self, data):
        """
        It cleans the dict to be easily read in the future.
        It crosses in every element of the dict in search of a list of dict to transform in a big dict:
        if there is a dict, it crosses every value (recalling this method).
        If there is a list, it crosses every item (recalling this method). then if the items are dicts the list
        is swapped with a big dict
        Args:
            data: it's the data contained in the yaml file as a dict

        Returns:
            data: it returns data cleaned, every list of dict is transformed in a single dict

        """
        # using yaml there is a problem:
        # it has no strict rules, so you can have [[{}]] [[]] {[]} {{}} ecc
        # this recursive method transform everything as {...{}...} or {...[]...}
        temp_dict = {}
        if isinstance(data, dict):
            for key in data.keys():
                # the model dict follow a particular configuration that is necessary not to change
                if key != 'model':
                    value = self.__clean_dict(data[key])
                    data.update({key: value})
        if isinstance(data, list):
            for element in data:
                element = self.__clean_dict(element)
                # the following code follow a statement that is always true using yaml:
                # if in the list one element is a dict, so are all the others elements
                if isinstance(element, dict):
                    temp_dict.update(element)
        if bool(temp_dict):
            data = temp_dict
        return data


    def get_parameters(self):
        """
        Get the extraction configurations.

        Returns:
            dict: A dictionary containing extraction configurations for visual, textual, and visual_textual data.
        """
        # experiments_dict = {key: copy.deepcopy(self._data_dict[key]) for key in ['visual', 'textual', 'visual_textual'] if key in self._data_dict}
        experiments_dict = {key: copy.deepcopy(self._param_data_dict[key]) for key in ['databases', 'preprocessing', 'feature_selection', 'training'] if key in self._param_data_dict}
        return experiments_dict




# Configure custom loguru levels
logger.configure(
    levels=[dict(name="NEW", no=13, icon="¤", color=""), dict(name="WELCOME", no=25, color="<green>", icon="!!!")],
)

# Hide Torch warnings
warnings.filterwarnings("ignore")