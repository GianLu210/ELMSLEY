from ..father_classes.YamlFileManagerFather import YamlFileManagerFather
import yaml
import importlib


class YamlFileManagerModels(YamlFileManagerFather):

    def __init__(self, yaml_file_path = 'config/modelMap.yml'):
        self.model_yaml_file_path = yaml_file_path
        super().__init__(yaml_file_path)


    def load_models_from_yaml(self):
        with open(self.model_yaml_file_path, "r") as file:
            config = yaml.safe_load(file)

        models = []
        for model_info in config["models"]:
            module = importlib.import_module(model_info["module"])
            model_class = getattr(module, model_info["class"])
            model = model_class(**model_info.get("params", {}))
            models.append(model)

        return models