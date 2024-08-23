import numpy as np
import pandas as pd
import os.path as op
import pickle
from scipy.fft import fft, fftfreq

def load_obj(name, path):
    with open(op.join(path, name), 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name, path):
    with open(op.join(path, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

data_path = "/sps/crnl/pmouches/data/MEG_PHRC_2006_Ap18"
file_prefix = "data_raw_"

patients = ["25", "27", "38", "61", "83", "97", "18", "22", "54", "68", "69", "84", "105", "42", "20", "30", "11", "47", "56", "101", "117", "4", "8", "26", "66", "67", "79", "92", "111", "2", "24", "65", "73", "100", "118", "119", "23", "35","40", "62", "102", "107", "108", "109"]

def create_windows_dataframe(data, samples_per_window, nb_samples, nb_channels, sample_rate, overlap_samples):
    windows = []
    
    for i in range(len(data)):
        start = 0
        end = samples_per_window
        while end < nb_samples:
            window_data = {"is_annoted": False, "block_id":i}
            for k in range(nb_channels):
                channel_data = data[i][k, start:end]
                window_data[f"Channel_{k+1}"] = np.array(channel_data)
            window_data["Start_Time"] = start / sample_rate 
            window_data["End_Time"] = end / sample_rate  
            windows.append(window_data)
            start += samples_per_window - overlap_samples
            end = start + samples_per_window
    df = pd.DataFrame(windows)
    return df

def annotate_window(annotations, df):
    sample_rate = 150
    for i in range(len(annotations)):
        for event in annotations[i]:
            time_stamp = event/sample_rate
            for index, row in df.iterrows():
                if row["Start_Time"] <= time_stamp <= row["End_Time"] and row["block_id"] == i:
                    df.at[index, "is_annoted"] = True
                    
    return df

def window_creation(data):
    data = data[data['is_reannoted'] == True]
    data.reset_index(drop=True, inplace=True)
    nb_channels = data["meg"][0].shape[0]
    nb_samples = data["meg"][0].shape[1]
    sample_rate = 150 # Hz
    window_size = 0.2 # 200 ms
    overlap = 0.03 # 30 ms

    samples_per_window = int(window_size * sample_rate)
    overlap_samples = int (overlap * sample_rate)
    
    df = create_windows_dataframe(data["meg"], samples_per_window, nb_samples,nb_channels, sample_rate, overlap_samples)
    df = annotate_window(data["reannot"], df)
    return df

def compute_window_ppa(window):
    return np.max(window) - np.min(window)

def compute_window_upslope(window):
    return np.max(np.diff(window))

def compute_window_downslope(window):
    return np.min(np.diff(window))

def compute_window_std(window):
    return np.std(window)

def compute_window_amplitude_ratio(window):
    return (np.max(window)-np.min(window))/np.mean(window)

def compute_window_sharpness(window):
    slopes = np.diff(window)
    return np.max(np.abs(slopes[1:]-slopes[:-1]))

def compute_window_average_slope(window):
    abs_slopes = np.abs(np.diff(window))
    return np.max((abs_slopes[:-1] + abs_slopes[1:]) / 2)

def compute_window_main_frequency(window):
    n = len(window)
    fft_result = fft(window)
    frequencies = fftfreq(n)
    amplitudes = np.abs(fft_result)
    peak_frequency_index = np.argmax(amplitudes)
    main_frequency = frequencies[peak_frequency_index]
    return main_frequency

def compute_window_phase_congruency(window):
    fft_window = fft(window)
    phases = np.angle(fft_window)
    phase_diff = np.diff(phases)
    phase_congruencies = 1 - (np.abs(phase_diff)/np.pi)
    return np.max(phase_congruencies)

list_of_features = ["ppa", "upslope", "std", "amplitude_ratio", "average_slope"]
compute_features = {"ppa" : compute_window_ppa, "upslope" : compute_window_upslope, "downslope": compute_window_downslope, "std": compute_window_std, 
                    "amplitude_ratio": compute_window_amplitude_ratio, "sharpness" : compute_window_sharpness, "average_slope": compute_window_average_slope, "main_frequency" : compute_window_main_frequency,
                    "phase_congruency": compute_window_phase_congruency}

def create_features(df):
    channels = [col for col in df.columns if col.startswith("Channel_")]    
    data_to_df = {}
    for channel in channels:
        channel_features = {feat: [] for feat in list_of_features}
        for window in df[channel]:
            for feat in list_of_features:
                channel_features[feat].append(compute_features[feat](window))
        for feat in list_of_features:
            data_to_df[f'{feat}_{channel}'] = channel_features[feat]
    return pd.DataFrame(data_to_df)

from sklearn.preprocessing import  MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2


def standardization(df):
    channels = [col for col in df.columns if col.startswith("Channel_")]    
    for channel in channels:
        channel_data = np.array(df[channel].tolist())
        samples_per_window = channel_data.shape[1]
        channel_data = channel_data.reshape(-1)
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        standardized_data = (channel_data - mean) / std
        channel_data = standardized_data.reshape(-1, samples_per_window)
        df[channel] = channel_data.tolist()
    return df

def preprocessing(set):
    y = set['is_annoted']
    standardized_set = standardization(set)
    X = create_features(standardized_set)
    min_max_scaler = MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled_df, y

def balance_labels(X, y):
    # Concaténer X et y pour pouvoir les manipuler ensemble
    df = pd.concat([X, y], axis=1)

    # Séparer les lignes avec label True et False
    df_true = df[df[y.name] == True]
    df_false = df[df[y.name] == False]

    # Sous-échantillonner les lignes avec label False
    df_false_sampled = df_false.sample(n=len(df_true), random_state=42)

    # Combiner les deux sous-ensembles
    df_balanced = pd.concat([df_true, df_false_sampled])

    # Mélanger les lignes pour plus d'aléatoire
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Séparer à nouveau les features et les labels
    X_balanced = df_balanced.drop(columns=[y.name])
    y_balanced = df_balanced[y.name]

    return X_balanced, y_balanced

# Patient splitting
cv = 5
patients_per_fold = len(patients) // cv

patients_df = {}
for patient in patients:
    data = pd.DataFrame(load_obj(f"{file_prefix}{patient}.pkl",data_path))
    patient_df = window_creation(data)
    X, y = preprocessing(patient_df)
    patients_df[f"X_{patient}"] = X
    patients_df[f"y_{patient}"] = y
    
file_name = "balanced_preprocessed_data"
path = "/sps/crnl/ademasson/data/features_extraction"
save_obj(patients_df, file_name, path)


# for i in range(0,cv):
#     test_patients = patients[i*patients_per_fold:(i+1)*patients_per_fold]
#     train_patients = patients[:i*patients_per_fold] + patients[(i+1)*patients_per_fold:]
#     # Train sets
#     X_train = None
#     y_train = None
#     for j, patient in enumerate(train_patients):
#         X_patient_set = patients_df[f"X_{patient}"]
#         y_patient_set = patients_df[f"y_{patient}"]
#         if X_train is None:
#             X_train = X_patient_set
#         else:
#             X_train = pd.concat([X_train, X_patient_set], axis=0, ignore_index=True)

#         if y_train is None:
#             y_train = y_patient_set
#         else:
#             y_train = np.hstack([y_train, y_patient_set])
            
#     # Test sets
#     X_test = None
#     y_test = None
#     for j, patient in enumerate(test_patients):
#         X_patient_set = patients_df[f"X_{patient}"]
#         y_patient_set = patients_df[f"y_{patient}"]             
#         if X_test is None:
#             X_test = X_patient_set
#         else:
#             X_test = pd.concat([X_test, X_patient_set], axis=0, ignore_index=True)

#         if y_train is None:
#             y_test = y_patient_set
#         else:
#             y_test = np.hstack([y_test, y_patient_set])
            
    
#     # SelectKBest
#     train_selector = SelectKBest(chi2, k=X_train.shape[1])
#     y_train[:] = y_train[:X_train.shape[1]]
#     train_selector.fit_transform(X_train, y_train)
#     train_feature_scores = {feature: train_selector.scores_[i] for i, feature in enumerate(X_train.columns)}
#     train_feature_mean_scores = {}
#     for feature in list_of_features:
#         scores = []
#         for col in X_train.columns:
#             if col.startswith(feature):
#                 scores.append(train_feature_scores[col])
#         train_feature_mean_scores[feature] = np.mean(scores)
#     train_feature_mean_scores = {k: v for k, v in sorted(train_feature_mean_scores.items(), key=lambda x: x[1], reverse=True)}
    
#     test_selector = SelectKBest(chi2, k=X_test.shape[1])
#     y_test[:] = y_test[:X_test.shape[1]]
#     test_selector.fit_transform(X_test, y_test)
#     test_feature_scores = {feature: test_selector.scores_[i] for i, feature in enumerate(X_test.columns)}
#     test_feature_mean_scores = {}
#     for feature in list_of_features:
#         scores = []
#         for col in X_test.columns:
#             if col.startswith(feature):
#                 scores.append(test_feature_scores[col])
#         test_feature_mean_scores[feature] = np.mean(scores)
#     test_feature_mean_scores = {k: v for k, v in sorted(test_feature_mean_scores.items(), key=lambda x: x[1], reverse=True)}
    
#     print("--------------------------")
#     print(f"CV n° {i+1}")
#     print("--------------------------")
#     print("Feature scores on Train set : ")
#     for k,v in train_feature_mean_scores:
#         print(k + " : " + v)
#     print("--------------------------")
#     print("Feature scores on Test set : ")
#     for k,v in train_feature_mean_scores:
#         print(k + " : " + v)
#     print("--------------------------")
    
