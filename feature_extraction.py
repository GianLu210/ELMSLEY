import pickle
import pandas as pd
import numpy as np
import os
from math import degrees
import tqdm


def slope(v1, v2):
    return (v2[1] - v1[1]) / (v2[0] - v1[0])


def ang(P1, P2, P3):
    v1 = (P2[0] - P1[0], P2[1] - P1[1])
    v2 = (P3[0] - P2[0], P3[1] - P2[1])

    return degrees(np.arccos((np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))))


def distance_ms(v1, v2, fs):
    return (v2[0] - v1[0]) / fs


complete_db = pd.DataFrame(columns=['dataset', 'sample', 'P_seg', 'PQ_seg', 'QRS_seg', 'QR_seg',
                                    'QT_seg', 'RS_seg', 'ST_seg', 'T_seg', 'PT_seg', 'PonPQ_ang',
                                    'RoffQR_ang', 'QRS_ang', 'RST_ang', 'STToff_ang', 'PQ_slope',
                                    'QR_slope', 'RS_slope', 'ST_slope', 'QR_to_QS', 'RS_to_QS', 'P_Peak', 'Q_Peak',
                                    'R_Peak', 'S_Peak', 'T_Peak', 'RR_Back', 'RR_Forw', 'RR_mean', 'ECG_label'])


def feature_extraction(db_dict, pre_proc_dir='pre_processed', complete_dataset_name='complete_dataset'):
    for db_name in db_dict.keys():
        files = os.listdir(os.path.join(pre_proc_dir, db_name))
        files = [os.path.splitext(os.path.basename(file_name))[0]
                 for file_name in files
                 if os.path.splitext(file_name)[-1] in {'.pickle'}]

        for file in tqdm.tqdm(files, desc=f'Extracting features...'):
            with open(f"{os.path.join(pre_proc_dir, db_name, file)}.pickle", 'rb') as pickle_file:
                seg_pickle = pickle.load(pickle_file)
            label = os.path.splitext(file)[0].split('_')[-1]
            fs = seg_pickle['fs']
            seg = seg_pickle['seg_df']
            P_peak = seg.loc['values', 'ECG_P_Peaks']
            Q_peak = seg.loc['values', 'ECG_Q_Peaks']
            R_peak = seg.loc['values', 'ECG_R_Peaks']
            S_peak = seg.loc['values', 'ECG_S_Peaks']
            T_peak = seg.loc['values', 'ECG_T_Peaks']

            P_on = (seg.loc['index', 'ECG_P_Onsets'], seg.loc['values', 'ECG_P_Onsets'])
            P_off = (seg.loc['index', 'ECG_P_Offsets'], seg.loc['values', 'ECG_P_Offsets'])
            P = (seg.loc['index', 'ECG_P_Peaks'], seg.loc['values', 'ECG_P_Peaks'])
            Q = (seg.loc['index', 'ECG_Q_Peaks'], seg.loc['values', 'ECG_Q_Peaks'])
            R_on = (seg.loc['index', 'ECG_R_Onsets'], seg.loc['values', 'ECG_R_Onsets'])
            R_off = (seg.loc['index', 'ECG_R_Offsets'], seg.loc['values', 'ECG_R_Offsets'])
            R = (seg.loc['index', 'ECG_R_Peaks'], seg.loc['values', 'ECG_R_Peaks'])
            S = (seg.loc['index', 'ECG_S_Peaks'], seg.loc['values', 'ECG_S_Peaks'])
            T_on = (seg.loc['index', 'ECG_T_Onsets'], seg.loc['values', 'ECG_T_Onsets'])
            T_off = (seg.loc['index', 'ECG_T_Offsets'], seg.loc['values', 'ECG_T_Offsets'])
            T = (seg.loc['index', 'ECG_T_Peaks'], seg.loc['values', 'ECG_T_Peaks'])

            P_seg = distance_ms(P_on, P_off, fs)
            PQ_seg = distance_ms(P_on, Q, fs)
            QRS_seg = distance_ms(R_on, R_off, fs)
            QR_seg = distance_ms(Q, R, fs)
            QT_seg = distance_ms(Q, T, fs)
            RS_seg = distance_ms(R, S, fs)
            ST_seg = distance_ms(S, T, fs)
            T_seg = distance_ms(T_on, T_off, fs)
            PT_seg = distance_ms(P_on, T_off, fs)
            QS_seg = distance_ms(Q, S, fs)

            PonPQ_ang = ang(P_on, P, Q)
            PoffQR_ang = ang(P_off, Q, R)
            QRS_ang = ang(Q, R, S)
            RST_ang = ang(R, S, T)
            STToff_ang = ang(S, T, T_off)

            PQ_slope = slope(P, Q)
            QR_slope = slope(Q, R)
            RS_slope = slope(R, S)
            ST_slope = slope(S, T)

            QR_on_QS = QR_seg / QS_seg
            RS_on_QS = RS_seg / QS_seg

            RR_forw = seg.loc['ECG_RR_Forw', 'ECG_R_Peaks'] / fs
            RR_back = seg.loc['ECG_RR_Back', 'ECG_R_Peaks'] / fs
            RR_mean = seg.loc['ECG_RR_mean', 'ECG_R_Peaks'] / fs

            complete_db.loc[len(complete_db)] = [db_name, os.path.splitext(file)[0].split('_')[0], P_seg, PQ_seg, QRS_seg,
                                                 QR_seg,
                                                 QT_seg, RS_seg, ST_seg, T_seg, PT_seg, PonPQ_ang,
                                                 PoffQR_ang, QRS_ang, RST_ang, STToff_ang,
                                                 PQ_slope, QR_slope, RS_slope, ST_slope,
                                                 QR_on_QS, RS_on_QS, P_peak, Q_peak,
                                                 R_peak, S_peak, T_peak, RR_back, RR_forw, RR_mean, label]

    with open(f'{complete_dataset_name}.pickle', 'wb') as complete_data:
        pickle.dump(complete_db, complete_data)
    print('Extraction complete')
