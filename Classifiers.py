import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, learning_curve, GroupKFold
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from utilities import elbow_method
from math import ceil
import os
from sklearn.cluster import KMeans


class Training_obj():

    def __init__(self, data, selected_features=None, test_size=0.3, min_samples_per_patient=None, oversample=False, undersample=False,
                 oversampling_method='ADASYN', cascade=False, inter_patient=False, cross_dataset=False,
                 train_datasets=None, test_datasets=None, excluded_labels=[], results_folder='results',
                 use_grid_search=False, scale_features=False, explain=False):
        self.data = data
        self.selected_features = selected_features
        self.test_size = test_size
        self.min_samples_per_patient = min_samples_per_patient
        self.oversample = oversample
        self.undersample = undersample
        self.oversampling_method = oversampling_method
        self.cascade = cascade
        self.excluded_labels = excluded_labels + ['None']
        self.results_folder = results_folder
        self.use_grid_search = use_grid_search
        self.label_encoder = LabelEncoder()
        self.scale_features = scale_features  # Flag per abilitare lo scaling
        self.scaler = None
        self.explain = explain
        self.inter_patient = inter_patient
        self.cross_dataset = cross_dataset
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.external_test = None

    def get_true_negative(self, best_model, x_test, y_test):
        y_pred = best_model.predict(x_test)
        tn_mask = (y_pred == 1) & (y_test == 1)
        x_tn = x_test[tn_mask]
        return x_tn

    def _train_once(self, x_train, y_train, x_test, y_test, x_external, y_external, label_encoder, model_name,
                    model_config, stage_name="default"):
        #models = self._define_models(x_train.shape[1])
        print(f"Training {model_name}...")
        best_model = self._train_model(x_train, y_train, model_config["model"], model_config["param_grid"])
        self._evaluate_model(best_model, x_train, y_train, x_test, y_test, x_external, y_external, label_encoder,
                             model_name, stage_name)
        return best_model


    def _define_models(self, n_features):
        return {
            "LogisticRegression": {
                "model": LogisticRegression(max_iter=10000, random_state=42),
                "param_grid": {
                    "C": [1, 0.8, 0.5],
                    "solver": ["newton-cg", "sag", "saga", "lbfgs"]
                }
             },
            "RandomForest": {
                "model": RandomForestClassifier(random_state=42),
                "param_grid": {
                    "n_estimators": np.arange(50, 201, 50),
                    "max_features": np.arange(1, n_features),
                    "max_depth": [4, 6, 8]
                }
             },
            "XGBoost": {
                "model": XGBClassifier(random_state=42),
                "param_grid": {
                    "n_estimators": np.arange(50, 201, 50),
                    "max_features": np.arange(1, n_features),
                    "max_depth": [4, 6, 8],
                    "learning_rate": [0.001, 0.01, 0.1, 1]
                }
            },
            "LightGBM": {
                "model": LGBMClassifier(random_state=42),
                "param_grid": {
                    "n_estimators": np.arange(50, 201, 50),
                    "max_features": np.arange(1, n_features),
                    "max_depth": [4, 6, 8],
                    "learning_rate": [0.001, 0.01, 0.1, 1]
                }
            },
            "MLPClassifier": {
                "model": MLPClassifier(random_state=42),
                "param_grid": {
                    "hidden_layer_sizes": [(100,)],
                    "activation": ["logistic", "relu"],
                    "solver": ["adam", "sgd"],
                    "batch_size": [256, 512],
                    "learning_rate_init": [0.001, 0.01, 0.1],
                    "max_iter": [200, 400, 800]
                }
            },
            "SVM": {
                "model": SVC(random_state=42, probability=True),
                "param_grid": {
                    "C": [0.5, 0.8, 1.0],
                    "kernel": ["rbf", "linear", "sigmoid"]
                }
            }
        }

    def _train_model(self, x_train, y_train, model, param_grid):

        if self.use_grid_search:
            print(f"Eseguendo Grid Search per {model.__class__.__name__}...")
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=3,
                                       scoring="balanced_accuracy")
            grid_search.fit(x_train, y_train)
            print(f"Best parameters: {grid_search.best_params_}")
            return grid_search.best_estimator_
        else:
            print(f"Addestramento senza Grid Search per {model.__class__.__name__}...")
            model.fit(x_train, y_train)  # Addestra il modello con i parametri di default
            return model

    def splitting(self, min_test_size=0.1):

        if self.selected_features is not None:
            self.data = self.data[['dataset', 'sample', 'ECG_label'] + self.selected_features]

        if self.min_samples_per_patient is not None:
            counts = self.data['sample'].value_counts()
            valid_patients = counts[counts >= self.min_samples_per_patient].index
            self.data = self.data[self.data['sample'].isin(valid_patients)]

        if self.excluded_labels is not None:
            self.data = self.data[~self.data['ECG_label'].isin(self.excluded_labels)]

        # Escludo Incart dal train
        if self.cross_dataset and self.train_datasets is not None or self.test_datasets is not None:
            if self.train_datasets is not None:
                train = self.data[self.data['dataset'].isin(self.train_datasets)]
                self.external_test = self.data[~self.data['dataset'].isin(self.train_datasets)]
            elif self.test_datasets is not None:
                train = self.data[~self.data['dataset'].isin(self.test_datasets)]
                self.external_test = self.data[self.data['dataset'].isin(self.test_datasets)]

        else:
            train = self.data


        grouped = train.groupby('sample')['ECG_label'].first().reset_index()

        # Uso GroupShuffleSplit per fare una suddivisione stratificata
        valid_splits = []
        valid_split = False
        while not valid_split and self.test_size >= 0.1:

            splitter = GroupShuffleSplit(n_splits=5, test_size=self.test_size, random_state=42)

            # Divido i gruppi di sample in train e test
            for train_idx, test_idx in splitter.split(grouped, grouped['ECG_label'], groups=grouped['sample']):
                train_ids = grouped.iloc[train_idx]['sample']
                test_ids = grouped.iloc[test_idx]['sample']

                # Creo il train set e il test set usando i sample separati
                train_df = train[train['sample'].isin(train_ids)]
                test_df = train[train['sample'].isin(test_ids)]

                train_size = len(train_df)
                test_size_actual = len(test_df)
                total_size = train_size + test_size_actual

                train_ratio = ceil((train_size / total_size)*100)/100
                test_ratio = ceil((test_size_actual / total_size)*100)/100

                if 0.6 <= train_ratio <= 0.7 and test_ratio >= min_test_size:
                    valid_split = True
                    valid_splits.append((train_df, test_df))
                    print(f'Best splitting obtained for split number {len(valid_splits)}: train:{train_ratio}; test: {test_ratio}')
                    break
                else:
                    self.test_size = self.test_size - 0.01


        return valid_splits


    def train(self):

        splittings = self.splitting()
        shape = splittings[0][0].shape[1]

        #models = self._define_models(self.x_train.shape[1])
        models = self._define_models(shape)
        for model_name, model_config in models.items():
            for split in splittings:
                self._preprocess(split)

                if self.cascade:
                    print('Performing two steps classification...')
                    print('N vs rest')
                    # Prima fase: N vs rest
                    y_train_N_vs_rest = np.where(self.y_train == self.label_encoder.transform(['N']), 0, 1)
                    y_test_N_vs_rest = np.where(self.y_test == self.label_encoder.transform(['N']), 0, 1)
                    if self.cross_dataset:
                        y_external_N_vs_rest = np.where(self.y_external == self.label_encoder.transform(['N']), 0, 1)
                    else:
                        y_external_N_vs_rest = []

                    N_vs_rest_model = self._train_once(self.x_train, y_train_N_vs_rest, self.x_test, y_test_N_vs_rest, self.x_external,
                                     y_external_N_vs_rest, self.label_encoder, model_name, model_config, stage_name="N_vs_rest")

                    mask_VEB_vs_SVEB_train = self.y_train != self.label_encoder.transform(['N'])
                    mask_VEB_vs_SVEB_test = self.y_test != self.label_encoder.transform(['N'])


                    y_train_filtered = self.y_train[mask_VEB_vs_SVEB_train]
                    x_train_VEB_vs_SVEB = self.x_train[mask_VEB_vs_SVEB_train]
                    y_test_filtered = self.y_test[mask_VEB_vs_SVEB_test]
                    x_test_VEB_vs_SVEB = self.x_test[mask_VEB_vs_SVEB_test]




                    # Rifittare il LabelEncoder con le sole classi VEB e SVEB
                    le_veb_sveb = LabelEncoder()
                    le_veb_sveb.fit(['VEB', 'SVEB'])

                    # Binarizzare le classi
                    y_train_VEB_vs_SVEB = le_veb_sveb.transform(self.label_encoder.inverse_transform(y_train_filtered))
                    y_test_VEB_vs_SVEB = le_veb_sveb.transform(self.label_encoder.inverse_transform(y_test_filtered))


                    if self.cross_dataset:
                        mask_VEB_vs_SVEB_external = self.y_external != self.label_encoder.transform(['N'])
                        y_external_filtered = self.y_external[mask_VEB_vs_SVEB_external]
                        x_external_VEB_vs_SVEB = self.x_external[mask_VEB_vs_SVEB_external]
                        y_external_VEB_vs_SVEB = le_veb_sveb.transform(self.label_encoder.inverse_transform(y_external_filtered))
                    else:
                        x_external_VEB_vs_SVEB = []
                        y_external_VEB_vs_SVEB = []


                    print('VEB vs SVEB')

                    VEB_vs_SVEB_model = self._train_once(x_train_VEB_vs_SVEB, y_train_VEB_vs_SVEB, x_test_VEB_vs_SVEB, y_test_VEB_vs_SVEB,
                                     x_external_VEB_vs_SVEB, y_external_VEB_vs_SVEB, le_veb_sveb, model_name, model_config,
                                     stage_name="VEB_vs_SVEB")


                    print()
                else:
                    self._train_once(self.x_train, self.y_train, self.x_test, self.y_test, self.x_external,
                                     self.y_external, self.label_encoder, model_name, model_config)

    def _resample(self, x_train, y_train):
        if self.oversample:
            if self.oversampling_method == 'ADASYN':
                print('Oversampling with ADASYN algorithm started...')
                sampler = ADASYN(random_state=42, sampling_strategy='not majority')
            elif self.oversampling_method == 'SMOTE':
                print('Oversampling with SMOTE algorithm...')
                sampler = SMOTE(random_state=42, sampling_strategy='not majority')
            else:
                print('Oversampling method unspecified or not implemented')
                return x_train, y_train
        elif self.undersample:
            print('Random Undersampling algorithm...')
            sampler = RandomUnderSampler(random_state=42)
        else:
            return x_train, y_train  # No resampling

        return sampler.fit_resample(x_train, y_train)

    def _evaluate_model(self, model, x_train, y_train, x_test, y_test, x_external, y_external, label_encoder,
                        model_name, stage_name):

        model_folder = os.path.join(self.results_folder, model_name)
        external_folder = os.path.join(self.results_folder, 'external_db', model_name)
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(external_folder, exist_ok=True)

        # Previsioni numeriche
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)


        # Decodifica delle etichette numeriche
        y_pred_train_decoded = label_encoder.inverse_transform(y_pred_train)
        y_pred_test_decoded = label_encoder.inverse_transform(y_pred_test)
        y_train_decoded = label_encoder.inverse_transform(y_train)
        y_test_decoded = label_encoder.inverse_transform(y_test)


        # Usa le etichette decodificate per i report
        print(f"{model_name} - Classification Report (Train):")
        print(classification_report(y_train_decoded, y_pred_train_decoded))
        print(f"{model_name} - Classification Report (Test):")
        print(classification_report(y_test_decoded, y_pred_test_decoded))
        print(f"Recall: {recall_score(y_test_decoded, y_pred_test_decoded, average='macro')}")
        print(f"Precision: {precision_score(y_test_decoded, y_pred_test_decoded, average='macro')}")
        print(f"F1 score: {f1_score(y_test_decoded, y_pred_test_decoded, average='macro')}")


        # Usa le etichette decodificate per salvare report e confusion matrix
        train_report = classification_report(y_train_decoded, y_pred_train_decoded, output_dict=True)
        test_report = classification_report(y_test_decoded, y_pred_test_decoded, output_dict=True)


        # # Learning Curves
        # print('Elaborating learning curves')
        # train_sizes, train_scores, test_scores = learning_curve(
        #     model, x_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1,
        #     train_sizes=np.linspace(0.1, 1.0, 10), random_state=42, verbose=3
        # )
        # train_mean = np.mean(train_scores, axis=1)
        # train_std = np.std(train_scores, axis=1)
        # test_mean = np.mean(test_scores, axis=1)
        # test_std = np.std(test_scores, axis=1)
        #
        # plt.figure(figsize=(10, 6))
        # plt.plot(train_sizes, train_mean, label="Training Score", color="blue", marker="o")
        # plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.2)
        # plt.plot(train_sizes, test_mean, label="Validation Score", color="orange", marker="s")
        # plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="orange", alpha=0.2)
        # plt.title(f"Learning Curves ({model_name}) - {stage_name}")
        # plt.xlabel("Training Examples")
        # plt.ylabel("Score")
        # plt.legend(loc="best")
        # plt.grid()
        # plt.savefig(f"{model_folder}/Learning_Curve_{stage_name}_{model_name}.pdf", bbox_inches="tight")
        # plt.close()


        # Convertiamo il report in un DataFrame e salviamo su file CSV
        train_report_df = pd.DataFrame(train_report).transpose()
        test_report_df = pd.DataFrame(test_report).transpose()



        train_report_df.to_csv(f"{model_folder}/{model_name}_{stage_name}_train_classification_report.csv")
        test_report_df.to_csv(f"{model_folder}/{model_name}_{stage_name}_test_classification_report.csv")



        # Generazione e salvataggio della Confusion Matrix
        print('Confusion matrix calculation')
        print(confusion_matrix(y_test, y_pred_test))
        cm_train = confusion_matrix(y_train, y_pred_train)
        cm_test = confusion_matrix(y_test, y_pred_test)


        cm_train_df = pd.DataFrame(cm_train, index=np.unique(y_train), columns=np.unique(y_train))
        cm_test_df = pd.DataFrame(cm_test, index=np.unique(y_test), columns=np.unique(y_test))

        cm_train_df.to_csv(f"{model_folder}/{model_name}_{stage_name}_train_confusion_matrix.csv")
        cm_test_df.to_csv(f"{model_folder}/{model_name}_{stage_name}_test_confusion_matrix.csv")

        if self.cross_dataset:

            y_external_decoded = label_encoder.inverse_transform(y_external)
            y_pred_external = model.predict(x_external)
            y_pred_external_decoded = label_encoder.inverse_transform(y_pred_external)
            print(f"{model_name} - Classification Report (External):")
            print(classification_report(y_external_decoded, y_pred_external_decoded))

            external_report = classification_report(y_external_decoded, y_pred_external_decoded, output_dict=True)

            external_report_df = pd.DataFrame(external_report).transpose()
            external_report_df.to_csv(f"{external_folder}/{model_name}_{stage_name}_external_classification_report.csv")
            cm_external = confusion_matrix(y_external, y_pred_external)
            cm_external_df = pd.DataFrame(cm_external, index=np.unique(y_external), columns=np.unique(y_external))
            cm_external_df.to_csv(f"{external_folder}/{model_name}_{stage_name}_external_confusion_matrix.csv")

        if self.explain:

            # SHAP analysis
            print('Start explanation')
            k_optimal = elbow_method(x_train)
            try:
                explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(x_train, k_optimal))
            except:
                explainer = shap.TreeExplainer(model)

            #Estraggo subste dal test_set e dal dataset esterno per velocizzare il calcolo degli shap

            # shap_values = explainer.shap_values(x_test)
            #Uso il clastering sul train set per selezionare campioni significativi
            kmeans = KMeans(n_clusters=k_optimal, random_state=42)
            kmeans.fit(x_train)

            #Estraggo il subset dal test_set

            clusters = kmeans.predict(x_test)

            x_test['cluster'] = clusters
            min_subset_test = min(x_test['cluster'].value_counts())

            subset_test = x_test.groupby('cluster').apply(lambda x: x.sample(n=min_subset_test, random_state=42)).reset_index(drop=True)
            shap_values_ex = explainer(subset_test.drop(columns=['cluster']))

            #Estraggo un subset del dataset esterno

            clusters_external = kmeans.predict(x_external)
            x_external['cluster'] = clusters_external
            min_subset_external = min(x_external['cluster'].value_counts())
            subset_external = x_external.groupby('cluster').apply(lambda x: x.sample(n=min_subset_external, random_state=42)).reset_index(drop=True)


            #Estraggo gli output del subset di test per disegnare i waterfall plot

            subset_indices = subset_test.index
            y_subset_test = y_test_decoded[subset_indices]
            y_subset_pred = y_pred_test[subset_indices]

            #Estraggo gli output del subset del dataset esterno per disegnare i waterfall plot

            subset_external_indices = subset_external.index
            y_subset_external = y_external_decoded[subset_external_indices]
            y_subset_pred_external = y_pred_external[subset_external_indices]

            shap_values_external = explainer(subset_external.drop(columns=['cluster']))

            if stage_name == 'default':

                mean_0 = np.mean(np.abs(shap_values_ex.values[:, :, 0]), axis=0)
                mean_1 = np.mean(np.abs(shap_values_ex.values[:, :, 1]), axis=0)
                mean_2 = np.mean(np.abs(shap_values_ex.values[:, :, 2]), axis=0)
                df = pd.DataFrame({"NSR": mean_0, "VEB": mean_1, "SVEB": mean_2})

                mean_0_external = np.mean(np.abs(shap_values_external.values[:, :, 0]), axis=0)
                mean_1_external = np.mean(np.abs(shap_values_external.values[:, :, 1]), axis=0)
                mean_2_external = np.mean(np.abs(shap_values_external.values[:, :, 2]), axis=0)
                df_external = pd.DataFrame({"NSR": mean_0_external, "VEB": mean_1_external, "SVEB": mean_2_external})

            else:
                mean_0 = np.mean(np.abs(shap_values_ex.values[:, :, 0]), axis=0)
                mean_1 = np.mean(np.abs(shap_values_ex.values[:, :, 1]), axis=0)
                df = pd.DataFrame({f"{stage_name.split('_')[0]}": mean_0, f"{stage_name.split('_')[2]}": mean_1})

                mean_0_external = np.mean(np.abs(shap_values_external.values[:, :, 0]), axis=0)
                mean_1_external = np.mean(np.abs(shap_values_external.values[:, :, 1]), axis=0)
                df_external = pd.DataFrame({f"{stage_name.split('_')[0]}": mean_0_external, f"{stage_name.split('_')[2]}": mean_1_external})

            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            df.plot.bar(ax=ax)
            ax.set_ylabel("Means SHAP", size=30)
            ax.set_xticklabels(x_train.columns, rotation=45, size=20)
            ax.legend(fontsize=30)
            plt.savefig(f"{model_folder}/SHAP_Bar_{stage_name}_{model_name}.pdf",
                        bbox_inches="tight")
            plt.close()

            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            df_external.plot.bar(ax=ax)
            ax.set_ylabel("Means SHAP", size=30)
            ax.set_xticklabels(x_train.columns, rotation=45, size=20)
            ax.legend(fontsize=30)
            plt.savefig(f"{external_folder}/SHAP_Bar_{stage_name}_{model_name}.pdf",
                        bbox_inches="tight")
            plt.close()

            # Bar plots for correctly and incorrectly classified samples
            correctly_classified_idx = (y_subset_test == y_subset_pred)
            incorrectly_classified_idx = ~correctly_classified_idx

            if np.any(correctly_classified_idx):
                idx_correct = np.where(correctly_classified_idx)[0][0]
                for label in model.classes_:
                    shap.plots.waterfall(shap_values_ex[idx_correct, :, label], show=False)
                    plt.title("SHAP Values for Correctly Classified Sample")
                    plt.savefig(
                        f"{model_folder}/SHAP_Waterfall_Correct_{label_encoder.inverse_transform([label])[0]}_{stage_name}_{model_name}.pdf",
                        bbox_inches="tight")
                    plt.close()

            if np.any(incorrectly_classified_idx):
                idx_incorrect = np.where(incorrectly_classified_idx)[0][0]
                for label in model.classes_:
                    shap.plots.waterfall(shap_values_ex[idx_incorrect, :, label], show=False)
                    plt.title("SHAP Values for Incorrectly Classified Sample")
                    plt.savefig(
                        f"{model_folder}/SHAP_Waterfall_Incorrect_{label_encoder.inverse_transform([label])[0]}_{stage_name}_{model_name}.pdf",
                        bbox_inches="tight")
                    plt.close()

            # Bar plots external db for correctly and incorrectly classified samples

            correctly_classified_idx_external = (y_subset_external == y_subset_pred_external)
            incorrectly_classified_idx_external = ~correctly_classified_idx_external

            if np.any(correctly_classified_idx_external):
                idx_correct = np.where(correctly_classified_idx_external)[0][0]
                for label in model.classes_:
                    shap.plots.waterfall(shap_values_external[idx_correct, :, label], show=False)
                    plt.title("SHAP Values for Correctly Classified Sample External Database")
                    plt.savefig(
                        f"{external_folder}/SHAP_Waterfall_Correct_External_{label_encoder.inverse_transform([label])[0]}_{stage_name}_{model_name}.pdf",
                        bbox_inches="tight")
                    plt.close()

            if np.any(incorrectly_classified_idx_external):
                idx_incorrect = np.where(incorrectly_classified_idx_external)[0][0]
                for label in model.classes_:
                    shap.plots.waterfall(shap_values_external[idx_incorrect, :, label], show=False)
                    plt.title("SHAP Values for Incorrectly Classified Sample External Database")
                    plt.savefig(
                        f"{external_folder}/SHAP_Waterfall_Incorrect_{label_encoder.inverse_transform([label])[0]}_{stage_name}_{model_name}.pdf",
                        bbox_inches="tight")
                    plt.close()

    def LOSO(self, df, model):
        df = df[~df['sample'].str.contains('I')]
        #external_test = self.data[self.data['sample'].str.contains('I')]
        X = df.drop(columns=['ECG_label'])  # Caratteristiche
        y = df['ECG_label']  # Target
        groups = df['sample']  # Identificativo paziente

        filtered_patients = [id for id in df['sample'].unique() if len(df[df['sample'] == id]) >= 300]
        df = df[df['sample'].isin(filtered_patients)]


        gkf = GroupKFold(n_splits=len(df['sample'].unique()))  # Una fold per ogni paziente

        # Per salvare le metriche
        all_reports = []
        accuracies = []
        recalls= []
        precisions = []
        f1_scores = []
        models = []

        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            # Dividi train e test
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)  # Calcola media e std su train
            X_test_scaled = scaler.transform(X_test)



            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            report = classification_report(y_test, y_pred, output_dict=True)
            precisions.append(precision_score(y_test, y_pred, average='micro'))
            recalls.append(recall_score(y_test, y_pred, average='micro'))
            accuracies.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='micro'))
            models.append(model)
            #

            all_reports.append(report)

        mean_accuracy = np.mean(accuracies)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1_score = np.mean(f1_scores)

        print(f"Mean Accuracy: {mean_accuracy:.2f}")
        print(f"Mean Precision: {mean_precision:.2f}")
        print(f"Mean Recall: {mean_recall:.2f}")
        print(f"Mean F1: {mean_f1_score:.2f}")

        for i, report in enumerate(all_reports):
            print(f"Patient {i + 1} Classification Report:")
            print(report)

    def _split_external(self, external_test):
        x_external = external_test.drop(columns=['sample', 'ECG_label'])
        y_external = external_test['ECG_label']

        if self.scale_features:
            x_external = pd.DataFrame(self.scaler.transform(x_external), columns=x_external.columns)

        y_external_encoded = self.label_encoder.transform(y_external)

        self.x_external, self.y_external = x_external, y_external_encoded

    def _preprocess(self, split):
        train_df, test_df = split
        x_train = train_df.drop(columns=['sample', 'ECG_label'])
        y_train = train_df['ECG_label']

        x_test = test_df.drop(columns=['sample', 'ECG_label'])
        y_test = test_df['ECG_label']

        # Esegui il resampling
        x_train, y_train = self._resample(x_train, y_train)

        # Se lo scaling è abilitato, applica il scaler
        if self.scale_features:
            print('Starting data normalization')
            # self.scaler = StandardScaler()
            self.scaler = RobustScaler()
            x_train = pd.DataFrame(self.scaler.fit_transform(x_train), columns=x_train.columns)
            x_test = pd.DataFrame(self.scaler.transform(x_test), columns=x_test.columns)

            print('Normalization done!')
        # Codifica delle etichette
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        # Assegna le versioni codificate a self
        self.y_train = y_train_encoded
        self.y_test = y_test_encoded

        self.x_train = x_train
        self.x_test = x_test

        if self.cross_dataset:
            self._split_external(self.external_test)