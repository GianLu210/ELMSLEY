import os
import neurokit2 as nk
import wfdb
import numpy as np
import pickle
import pandas as pd
from math import floor

AAMI_dict = {'N': ['N', 'L', 'R', 'e', 'j'], 'SVEB': ['A', 'a', 'J', 'S'],
              'VEB': ['V', 'E'], 'F': ['F'], 'Q': ['/', 'f', 'Q', '+']}

class Preprocess:

    def __init__(self, db_dict, pre_proc_dir='pre_processed', rec_len=30, filtering_method='vg', label_dict=None):
        '''

        :param db_dict: {'MIT-ARR': {'db_dir': 'data_directory_path', 'ann_ext': 'annotation_extension', 'derivation': 'MLII'},
                         'MIT-NSR': {'db_dir': 'data_directory_path', 'ann_ext': 'annotation_extension', 'derivation': 'ECG1'}}
        :param pre_proc_dir:
        :param filtering_method: one of ['neurokit', 'biosppy', 'pantompkins1985', 'hamilton2002', 'elgendi2010', 'engzeemod2012', 'vg']
        :param derivation: the derivation to be used as signal
        :param label_dict: mapping of label in the annotation
        '''

        self.db_dict = db_dict
        self.pre_proc_dir = pre_proc_dir
        self.filtering_method = filtering_method

        self.rec_len = rec_len
        if label_dict is None:
            self.label_dict = AAMI_dict
        else:
            self.label_dict = label_dict
        os.makedirs(self.pre_proc_dir, exist_ok=True)


    def run_preprocessing(self):
        for db_name, detail_dict in self.db_dict.items():
            os.makedirs(os.path.join(self.pre_proc_dir, db_name), exist_ok=True)
            patients = self.__load_patients(detail_dict)
            for patient in patients:
                record = wfdb.rdrecord(os.path.join(detail_dict['db_dir'], patient))
                ann = wfdb.rdann(os.path.join(detail_dict['db_dir'], patient), detail_dict['ann_ext'])
                derivation_idx = self.__get_derivation_idx(record.sig_name, detail_dict['derivation'])
                rec_dur = len(record.p_signal[:, derivation_idx])/(record.fs * 60)
                if floor(rec_dur) > self.rec_len:
                    self._split_record(db_name, patient, record, ann, derivation_idx)
                    self._pre_process(db_name, patient, split=True)
                else:
                    self._pre_process(db_name, patient, record=record, ann=ann, derivation_idx=derivation_idx, split=False)



    def _split_record(self, db_name, patient, record, ann, derivation_idx):

        ds_split_dir = os.path.join(self.pre_proc_dir, db_name, 'split')
        os.makedirs(ds_split_dir, exist_ok=True)
        os.makedirs(os.path.join(ds_split_dir, patient), exist_ok=True)
        print(f'Inizio splitting dei record per il dataset {db_name}')

        n_samples = record.fs * self.rec_len * 60
        sample = 0
        n_seg = 0
        seg = dict()
        while sample < len(record.p_signal[:, derivation_idx]):
            if sample + n_samples < len(record.p_signal[:, derivation_idx]):
                seg['p_signal'] = record.p_signal[sample:sample + n_samples, derivation_idx]
                seg['annotation'] = [s - sample for s in ann.sample if s >= sample and s < sample + n_samples]
                seg['symbol'] = [sy for sy, s in zip(ann.symbol, ann.sample) if
                                 s >= sample and s < sample + n_samples]
                seg['fs'] = record.fs
                sample += n_samples
                if seg['annotation'] and seg['symbol']:
                    with open(os.path.join(ds_split_dir, patient, f'{patient}_{n_seg}.pickle'), 'wb') as file:
                        pickle.dump(seg, file)
                    n_seg += 1
                    print(f'Segmento {n_seg} del paziente {patient} salvato!')

            else:
                seg['p_signal'] = record.p_signal[:, 0][sample:]
                seg['annotation'] = [s for s in ann.sample if s >= sample and s < len(record.p_signal[:, derivation_idx])]
                seg['symbol'] = ann.symbol[sample:]
                seg['fs'] = record.fs
                sample += n_samples
                if seg['annotation'] and seg['symbol']:
                    with open(os.path.join(ds_split_dir, patient, f'{patient}_{n_seg}.pickle'), 'wb') as file:
                        pickle.dump(seg, file)
                    print(f'Segmento {n_seg + 1} del paziente {patient} salvato!')


    def _pre_process(self, db_name, patient, record=None, ann=None, derivation_idx=None, split=False):

        print(f'Processing patient: {patient}')

        if split:
            segments = os.listdir(os.path.join(self.pre_proc_dir, db_name, 'split', patient))
            for seg_file in segments:
                with open(os.path.join(self.pre_proc_dir, db_name, 'split', patient, seg_file), 'rb') as pickle_dict:
                    seg = pickle.load(pickle_dict)
                p_signal = seg['p_signal']
                ann_sample = seg['annotation']
                ann_symbol = seg['symbol']
                fs = seg['fs']
                self._save_information(db_name, patient, p_signal, ann_sample, ann_symbol, fs)
        else:
            p_signal = record.p_signal[:, derivation_idx]
            ann_symbol = [s for i, s in enumerate(ann.symbol) if ann.sample[i] > 0]
            ann_sample = ann.sample[-len(ann.symbol):]
            fs = record.fs
            self._save_information(db_name, patient, p_signal, ann_sample, ann_symbol, fs)


    def __get_label(self, symbol):
        for key in self.label_dict.keys():
            if symbol in self.label_dict[key]:
                return key
        # Viene eseguito solo se il simbolo non è presente nel dizionario
        print(symbol)
        return 'None'

    def _save_information(self, db_name, patient, p_signal, ann_sample, ann_symbol, fs):
        seg_idx = 0
        print("Filtering Signal...")
        clean_ecg = nk.ecg_clean(p_signal, sampling_rate=fs,
                                 method=self.filtering_method)
        print("Start Segmentation...")
        peaks = nk.ecg_delineate(clean_ecg, ann_sample, sampling_rate=fs, method='dwt')
        print(f'Segmentazione del paziente {patient} Effettuata')

        peaks_df = peaks[0].copy()
        peaks_df['ECG_R_Peaks'] = np.zeros(len(peaks_df), dtype=np.int8)
        peaks_df['ECG_RR_Back'] = np.zeros(len(peaks_df), dtype=np.int16)
        peaks_df['ECG_RR_Forw'] = np.zeros(len(peaks_df), dtype=np.int16)
        peaks_df.loc[ann_sample, 'ECG_R_Peaks'] = 1
        peaks_df.loc[ann_sample[1:], 'ECG_RR_Back'] = np.ediff1d(ann_sample).astype(np.int16)
        peaks_df.loc[ann_sample[1:-1], 'ECG_RR_Forw'] = np.ediff1d(ann_sample)[1:].astype(np.int16)
        peaks_df['ECG_RR_mean'] = peaks_df[['ECG_RR_Forw', 'ECG_RR_Back']].mean(axis=1)

        peaks_df['peaks'] = peaks_df.iloc[:].sum(axis=1)
        peaks_df = peaks_df.drop(peaks_df[peaks_df['peaks'] == 0].index)
        change_col = ['ECG_P_Onsets', 'ECG_P_Peaks', 'ECG_P_Offsets', 'ECG_R_Onsets', 'ECG_Q_Peaks', 'ECG_R_Peaks',
                      'ECG_S_Peaks', 'ECG_R_Offsets', 'ECG_T_Onsets', 'ECG_T_Peaks', 'ECG_T_Offsets', 'ECG_RR_Back',
                      'ECG_RR_Forw', 'ECG_RR_mean', 'peaks']
        peaks_df = peaks_df.reindex(change_col, axis=1)
        peaks_df = peaks_df.reset_index()

        p_on_idx = peaks_df[peaks_df['ECG_P_Onsets'] == 1].index
        for idx in p_on_idx:
            temp = idx
            try:
                for p in range(1, 12):
                    if peaks_df.iloc[temp, p] == 1:
                        temp += 1
                        if p == 11:
                            seg = peaks_df.iloc[idx:idx + 11, :-1]
                            seg_df = pd.DataFrame(
                                [seg['index'].values, clean_ecg[seg['index'].values], seg['ECG_RR_Back'].values,
                                 seg['ECG_RR_Forw'].values, seg['ECG_RR_mean']],
                                columns=seg.columns[1:-3],
                                index=['index', 'values', 'ECG_RR_Back', 'ECG_RR_Forw', 'ECG_RR_mean'])
                            symbol = ann_symbol[np.where(ann_sample == seg_df.loc['index', 'ECG_R_Peaks'])[0][0]]

                            pickle_name = os.path.join(self.pre_proc_dir, db_name,
                                                       patient + f'_{seg_idx}' + f'_{self.__get_label(symbol)}.pickle')
                            with open(pickle_name, 'wb') as file:
                                pickle.dump({'fs': fs, 'seg_df': seg_df}, file)
                            seg_idx += 1
                            print(f'Segmento {seg_idx} salvato del paziente {patient}')
                    else:
                        break
            except Exception as e:
                print(f'Error occurred in patient {patient} at sample {idx}')
                print(e)



    @staticmethod
    def __get_derivation_idx(sig_name, derivation):
        try:
            idx = sig_name.index(derivation)
            return idx

        except ValueError:
            print("Selected derivation is not present")

    @staticmethod
    def __load_patients(detail_dict):
        files = os.listdir(detail_dict['db_dir'])
        return {os.path.splitext(os.path.basename(file_name))[0]
                for file_name in files
                if os.path.splitext(file_name)[-1] not in {'', '.txt'}}