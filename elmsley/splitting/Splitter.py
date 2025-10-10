from sklearn.model_selection import train_test_split, GroupShuffleSplit
from math import ceil
import pandas as pd


class Splitter:
    def __init__(self, test_size, selected_features=None, train_datasets=None, test_datasets=None):

        self.test_size = test_size
        self.selected_features = selected_features
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets



    def split_data(self, data, min_test_size=0.1):

        if self.selected_features is not None:
            data = data[['dataset', 'sample', 'ECG_label'] + self.selected_features]

        if self.train_datasets is not None or self.test_datasets is not None:
            if self.train_datasets is not None:
                train = data[data['dataset'].isin(self.train_datasets)]
                external_test = data[~data['dataset'].isin(self.train_datasets)]
            elif self.test_datasets is not None:
                train = data[~data['dataset'].isin(self.test_datasets)]
                external_test = data[data['dataset'].isin(self.test_datasets)]

        else:
            train = data
            external_test = []

        grouped = train.groupby('sample')['ECG_label'].first().reset_index()

        # Uso GroupShuffleSplit per fare una suddivisione stratificata
        valid_splits = []
        valid_split = False
        while not valid_split and self.test_size >= 0.1:

            valid_split, valid_splits = self._try_splitting(train, grouped, min_test_size=min_test_size)

            if not valid_split:
                self.test_size -= 0.05


        if not valid_splits:
            print("No valid split found for the selected train/test ratio for intra-patient evaluation.")
            print("Performing inter-patient splitting with selected train/test ratio")
            X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1],
                                                                random_state=42)
            train_df = pd.concat([X_train, y_train], axis=1, names=[X_train.columns, 'ECG_label'])
            test_df = pd.concat([X_test, y_test], axis=1, names=[X_train.columns, 'ECG_label'])
            valid_splits.append((train_df, test_df))

        return valid_splits, external_test



    def _try_splitting(self, train, grouped, min_test_size):
        valid_splits = []
        splitter = GroupShuffleSplit(n_splits=2, test_size=self.test_size, random_state=42)

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

            train_ratio = ceil((train_size / total_size) * 100) / 100
            test_ratio = ceil((test_size_actual / total_size) * 100) / 100

            if 0.6 <= train_ratio <= 0.7 and test_ratio >= min_test_size:
                valid_split = True
                valid_splits.append((train_df, test_df))
                print(
                    f'Best splitting obtained for split number {len(valid_splits)}: train:{train_ratio}; test: {test_ratio}')
                return valid_split, valid_splits

            else:
                valid_split = False
                valid_splits = []
                return valid_split, valid_splits

