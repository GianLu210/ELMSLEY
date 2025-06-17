from sklearn.feature_selection import RFECV
from sklearnex.ensemble import RandomForestClassifier
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, recall_score
import pandas as pd

class RFE:
    def __init__(self, complete_dataset_name='complete_dataset', estimator=RandomForestClassifier(random_state=42),
                scaler=RobustScaler()):

        self.estimator = estimator
        self.scaler = scaler
        with open(f'{complete_dataset_name}.pickle', 'rb') as file:
            self.df = pickle.load(file)


        # Estrai la variabile target 'ECG_label'


    def perform_RFE(self):

        y = self.df['ECG_label']

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        # Estrai le features X, escludendo la colonna 'ECG_label'
        X_original = self.df.drop(columns=['dataset', 'sample', 'ECG_label'])
        X_scaled = self.scaler.fit_transform(X_original)

        X = pd.DataFrame(X_scaled, columns=X_original.columns)

        #TODO: make scorer selectable

        scorer = make_scorer(recall_score, average='macro')

        rfecv_selector = RFECV(estimator=self.estimator, step=1, cv=StratifiedKFold(10),
                               scoring=scorer, verbose=3)

        rfecv_selector.fit(X, y)
        rfecv_selector.ranking_.tofile('ranking.xlsx', sep=',')

        with open('rfe_selector.pickle', 'wb') as rfecv_selector_file:
            pickle.dump(rfecv_selector, rfecv_selector_file)

        # Visualizza il numero ottimale di caratteristiche selezionate
        print(f'Best number of features: {rfecv_selector.n_features_}')

        # Seleziona le caratteristiche ottimali
        selected_features = X.columns[rfecv_selector.support_]
        print(f'Selected features: {selected_features}')

        return selected_features

    def perform_RFE_gridsearch(self, grid_search_params):

        y = self.df['ECG_label']
        label_encoder = LabelEncoder()

        y = label_encoder.fit_transform(y)

        # Estrai le features X, escludendo la colonna 'ECG_label'
        X_original = self.df.drop(columns=['dataset', 'sample', 'ECG_label'])
        X_scaled = self.scaler.fit_transform(X_original)

        X = pd.DataFrame(X_scaled, columns=X_original.columns)

        scorer = make_scorer(recall_score, average='macro')


        grid_estimator = GridSearchCV(estimator=self.estimator, param_grid=grid_search_params)

        rfecv_selector = RFECV(estimator=grid_estimator, step=1, cv=StratifiedKFold(10),
                               scoring=scorer, verbose=3)

        rfecv_selector.fit(X, y)
        rfecv_selector.ranking_.tofile('ranking.xlsx', sep=',')

        with open('rfe_selector.pickle', 'wb') as rfecv_selector_file:
            pickle.dump(rfecv_selector, rfecv_selector_file)

        # Visualizza il numero ottimale di caratteristiche selezionate
        print(f'Best number of features: {rfecv_selector.n_features_}')

        # Seleziona le caratteristiche ottimali
        selected_features = X.columns[rfecv_selector.support_]
        print(f'Selected features: {selected_features}')

        return selected_features
