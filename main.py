from Preprocess import Preprocess
from Classifiers import Training_obj
from feature_extraction import feature_extraction
from RFE import perform_RFE, perform_RFECV
import numpy as np
import pickle

# db_dict = {'MIT-ARR': {'db_dir': "D://ECG_Datasets//mit-bih-arrhythmia-database",
#                        'ann_ext': 'atr',
#                        'derivation': 'MLII'},
#            'MIT-NSR': {'db_dir': "D://ECG_Datasets//mit-bih-normal-sinus-rhythm-database",
#                        'ann_ext': 'atr',
#                        'derivation': 'ECG1'}}


db_dict = {'MIT-NSR': {'db_dir': "D://ECG_Datasets//mit-bih-normal-sinus-rhythm-database",
                       'ann_ext': 'atr',
                       'derivation': 'ECG1'}}


pre_proc_dir = 'pre_processed'
complete_dataset_name = 'complete_dataset'

pre_proc_obj = Preprocess(db_dict, pre_proc_dir=pre_proc_dir)
pre_proc_obj.run_preprocessing()

feature_extraction(db_dict, pre_proc_dir=pre_proc_dir, complete_dataset_name=complete_dataset_name)

selected_features = perform_RFE()

with open(f'{complete_dataset_name}.pickle', 'rb') as dataset:
    df = pickle.load(dataset)

#df = df[df['sample'].str.contains('I')]
df = df[['sample', 'ECG_label'] + selected_features]
df = df.replace(['None', 'F', 'Q'], np.nan)
df = df.dropna()

counts = df['sample'].value_counts()

valid_patients = counts[counts >= 300].index

df = df[df['sample'].isin(valid_patients)]

training_obj = Training_obj(df, oversample=False, undersample=False,
                            oversampling_method=None, inter_patient=True, cascade=False,
                            use_grid_search=False, scale_features=True, explain=False)
training_obj.train()

#load data

#filter data

#segment record

#beat splitting

#feature extraction

#RFE

#train

