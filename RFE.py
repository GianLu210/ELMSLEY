from sklearn.feature_selection import RFECV
from sklearnex.ensemble import RandomForestClassifier
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, recall_score
import pandas as pd


def perform_RFE(complete_dataset_name='complete_dataset', estimator=RandomForestClassifier(random_state=42),
                scaler=RobustScaler()):
    scaler = scaler
    with open(f'{complete_dataset_name}.pickle', 'rb') as file:
        df = pickle.load(file)

    df = df.replace(['None', 'F', 'Q'], np.nan)
    df = df.dropna()

    counts = df['sample'].value_counts()

    valid_patients = counts[counts >= 300].index

    df = df[df['sample'].isin(valid_patients)]

    # Rimuovi le righe con valori mancanti
    dataset_senza_missing = df.dropna()

    # Estrai la variabile target 'ECG_label'
    y = dataset_senza_missing['ECG_label']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Estrai le features X, escludendo la colonna 'ECG_label'
    X_original = dataset_senza_missing.drop(columns=['sample', 'ECG_label'])
    X_scaled = scaler.fit_transform(X_original)

    X = pd.DataFrame(X_scaled, columns=X_original.columns)

    scorer = make_scorer(recall_score, average='weighted')

    rfecv_selector = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(10),
                           scoring=scorer, verbose=3)

    rfecv_selector.fit(X, y)
    rfecv_selector.ranking_.tofile('ranking.csv', sep=',')

    with open('rfe_selector.pickle', 'wb') as rfecv_selector_file:
        pickle.dump(rfecv_selector, rfecv_selector_file)

    # Visualizza il numero ottimale di caratteristiche selezionate
    print(f'Numero ottimale di caratteristiche: {rfecv_selector.n_features_}')

    # Seleziona le caratteristiche ottimali
    selected_features = X.columns[rfecv_selector.support_]
    print(f'Caratteristiche selezionate: {selected_features}')

    return selected_features


def perform_RFECV(complete_dataset_name='complete_dataset', estimator=RandomForestClassifier(random_state=42),
                scaler=RobustScaler()):
    scaler = scaler
    # Carica il dataset
    with open(f'{complete_dataset_name}.pickle', 'rb') as file:
        dataset = pickle.load(file)

    # Rimuovi le righe con valori mancanti
    dataset_senza_missing = dataset.dropna()

    # Estrai la variabile target 'ECG_label'
    y = dataset_senza_missing['ECG_label']

    # Estrai le features X, escludendo la colonna 'ECG_label'
    X = dataset_senza_missing.drop(columns=['sample', 'ECG_label'])

    # Dividi i dati in set di addestramento, validazione e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                                      random_state=42)  # 60% addestramento, 20% validazione, 20% test

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Definisci la griglia dei parametri per il tuning di RandomForest
    parametri_rf = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [None, 10, 20],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4]
    }
    # Inizializza il selettore RFECV con la cross-validation
    rfecv_selector = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(5),
                           scoring='accuracy', verbose=3)

    # Inizializza la GridSearchCV con il modello RFECV e la griglia dei parametri
    grid_search_rf = GridSearchCV(estimator=rfecv_selector, param_grid=parametri_rf, cv=StratifiedKFold(5),
                                  scoring='accuracy', verbose=3)

    # Applica la GridSearchCV
    grid_search_rf.fit(X_train, y_train)

    # Ottieni il miglior modello RandomForest
    best_rf_model = grid_search_rf.best_estimator_

    # Visualizza il numero ottimale di caratteristiche selezionate
    print(f'Numero ottimale di caratteristiche: {best_rf_model.n_features_}')

    # Seleziona le caratteristiche ottimali
    selected_features = X_train.columns[best_rf_model.support_]
    print(f'Caratteristiche selezionate: {selected_features}')

    return selected_features
