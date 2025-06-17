import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, learning_curve, GroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
import shap
from utilities import elbow_method
from balancing.Balancer import Balancer
from splitting.Splitter import Splitter
import os
from sklearn.cluster import KMeans

class Trainer:
    def __init__(self, data, models, test_size=0.3, oversampling_method=None, undersample=False, inter_patient=False,
                 train_datasets=None, test_datasets=None, scaler=RobustScaler(), results_folder='results', grid_search=False, explain=False):

        self.data = data
        self.models = models
        self.test_size = test_size
        self.oversampling_method = oversampling_method
        self.undersample = undersample
        self.inter_patient = inter_patient
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.scaler = scaler
        self.label_encoder = LabelEncoder()
        self.results_folder = results_folder
        self.grid_search = grid_search
        self.explain = explain
        self.x_external = []
        self.y_external = []


    def _train_once(self, x_train, y_train, x_test, y_test, x_external, y_external, label_encoder, model_name,
                    model_config, stage_name="default"):
        #models = self._define_models(x_train.shape[1])
        print(f"Training {model_name}...")
        best_model = self._train_model(x_train, y_train, model_config["model"], model_config["param_grid"])
        self._evaluate_model(best_model, x_train, y_train, x_test, y_test, x_external, y_external, label_encoder,
                             model_name, stage_name)
        return best_model


    def _train_model(self, x_train, y_train, model, param_grid):

        if self.grid_search:
            print(f"Starting Grid Search for {model.__class__.__name__}...")
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=3,
                                       scoring="f1_macro")
            grid_search.fit(x_train, y_train)
            print(f"Best parameters: {grid_search.best_params_}")
            return grid_search.best_estimator_
        else:
            print(f"Training with no Grid Search for {model.__class__.__name__}...")
            model.fit(x_train, y_train)  # Addestra il modello con i parametri di default
            return model


    def train(self):

        splitter = Splitter(self.test_size, selected_features=None, train_datasets=None, test_datasets=None)

        splittings, external_test = splitter.split_data(self.data)
        shape = splittings[0][0].shape[1]


        # models = self._define_models(self.x_train.shape[1])
        models = self._define_models(shape)
        for n_split, split in enumerate(splittings):
            print(f'Split {n_split + 1} of {len(splittings)}')
            self._prepare_data(split, external_test)
            for model_name, model_config in models.items():
                self._train_once(self.x_train, self.y_train, self.x_test, self.y_test, self.x_external,
                                 self.y_external, self.label_encoder, model_name, model_config)




    def define_models(self, n_features):
        pass


    def _prepare_data(self, split, external_test):
        train_df, test_df = split
        x_train = train_df.drop(columns=['dataset', 'sample', 'ECG_label'])
        y_train = train_df['ECG_label']

        x_test = test_df.drop(columns=['dataset', 'sample', 'ECG_label'])
        y_test = test_df['ECG_label']

        # Esegui il resampling
        balancer = Balancer(oversampling_method=self.oversampling_method, undersample=self.undersample)
        x_train, y_train = balancer.resample(x_train, y_train)

        # Se lo scaling è abilitato, applica il scaler

        print('Starting data normalization')
        x_train = pd.DataFrame(self.scaler.fit_transform(x_train), columns=x_train.columns)
        x_test = pd.DataFrame(self.scaler.transform(x_test), columns=x_test.columns)

        print('Normalization done!')

        # Codifica delle etichette
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        # Assegna le versioni codificate a self
        self.x_train = x_train
        self.y_train = y_train_encoded

        self.x_test = x_test
        self.y_test = y_test_encoded

        if self.train_datasets is not None or self.test_datasets is not None:
            self._split_external(external_test)



    def _split_external(self, external_test):
        x_external = external_test.drop(columns=['dataset', 'sample', 'ECG_label'])
        y_external = external_test['ECG_label']

        x_external = pd.DataFrame(self.scaler.transform(x_external), columns=x_external.columns)

        y_external_encoded = self.label_encoder.transform(y_external)

        self.x_external, self.y_external = x_external, y_external_encoded