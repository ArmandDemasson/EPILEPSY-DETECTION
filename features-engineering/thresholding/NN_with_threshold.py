# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os.path as op
import pickle

def load_obj(name, path):
    with open(op.join(path, name), 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name, path):
    with open(op.join(path, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

patients = ["42", "83", "4", "8", "65"]
## Preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

FILTER = 0.25


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
    for i in range(len(annotations)):
        for event in annotations[i]:
            time_stamp = event/150
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

def search_threshold(feature, threshold=None):
    true_means = np.array(feature[feature['y'] == 1]['mean'])
    true_means_sorted = np.sort(true_means)
    
    number_of_true = 1 - FILTER


    threshold_index = round((1 -number_of_true)*len(true_means_sorted))
    threshold = true_means_sorted[threshold_index]
    if threshold is not None:
        print(f"Seuil trouvé: {threshold}")
    else:
        print("Aucun seuil trouvé qui satisfait les conditions.")
    return threshold

def filter_by_thresholding(X, y, threshold=None):
    ppa_df = X.filter(like='ppa_')
    up_dev_df = X.filter(like='up_derivative')
    def top_20_features(row):
        return row.nlargest(20)
    ppa = pd.DataFrame()
    ppa['mean'] = ppa_df.apply(top_20_features, axis=1).mean(axis=1)
    ppa['y'] = y
    up_dev = pd.DataFrame()
    up_dev['mean'] = up_dev_df.apply(top_20_features, axis=1).mean(axis=1)
    up_dev['y'] = y
    if threshold is None:
        ppa_threshold = search_threshold(ppa)
        up_dev_threshold = search_threshold(up_dev)
    else:
        ppa_threshold = threshold['ppa']
        up_dev_threshold = threshold['up_dev']
    index_ppa = ppa[ppa['mean'] > ppa_threshold].index
    index_up_dev = up_dev[up_dev['mean'] > up_dev_threshold].index
    indexes = index_ppa.union(index_up_dev)
    y = np.array(y)
    y.reshape((y.shape[0],1))
    return X.loc[indexes], y[indexes], {'ppa': ppa_threshold, 'up_dev': up_dev_threshold}

def create_features(df):
    data_to_df = {}
    channels = [col for col in df.columns if col.startswith("Channel_")]
    for channel in channels:
        channel_up_derivatives = []
        channel_down_derivatives = []
        channel_ppa = []
        channel_amplitude_ratio = []
        for window in df[channel]:
            window_derivative = np.diff(window)
            channel_ppa.append(np.max(window) - np.min(window))
            channel_amplitude_ratio.append((np.max(window)-np.min(window))/np.mean(window))
            channel_up_derivatives.append(np.max(window_derivative))
            channel_down_derivatives.append(np.min(window_derivative))
        data_to_df[f'up_derivative_{channel}'] = channel_up_derivatives
        data_to_df[f'down_derivative_{channel}'] = channel_down_derivatives
        data_to_df[f'ppa_{channel}'] = channel_ppa
        data_to_df[f'amplitude_ratio_{channel}'] = channel_amplitude_ratio
    return pd.DataFrame(data_to_df)

def preprocessing(set):
    y = set['is_annoted']
    standardized_set = standardization(set)
    X = create_features(standardized_set)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled_df, y 

from sklearn.pipeline import Pipeline
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
predictions = {}
for i, test_patient in enumerate(patients): 
    X_train = None
    y_train = None
    for j, patient in enumerate(patients):
        if i != j:
            data = pd.DataFrame(load_obj(f"data_raw_{patient}.pkl","/sps/crnl/pmouches/data/MEG_PHRC_2006_Ap18"))
            patient_df = window_creation(data)
            X_patient_set, y_patient_set = preprocessing(patient_df)            
            
            if X_train is None:
                X_train = X_patient_set
            else:
                X_train = pd.concat([X_train, X_patient_set], axis=0, ignore_index=True)
            
            if y_train is None:
                y_train = y_patient_set
            else:
                y_train = np.hstack([y_train, y_patient_set])
    
    print("Number of windows before thresholds on train : ", len(y_train))
    X_train, y_train, threshold = filter_by_thresholding(X=X_train, y=y_train)
    print("Number of windows after thresholds on train : ", len(y_train))
    tf.random.set_seed(0)
    model = keras.Sequential(
    [
        keras.Input(shape=(X_train.shape[-1],)),
        layers.Dense(10, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
    )
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                loss=keras.losses.BinaryFocalCrossentropy(),
                metrics=["accuracy"])

    history=model.fit(X_train, y_train, epochs=100, validation_split = 0.2, batch_size=32)

    ## Testing
    test_data = pd.DataFrame(load_obj(f"data_raw_{test_patient}.pkl","/sps/crnl/pmouches/data/MEG_PHRC_2006_Ap18"))
    test_df = window_creation(test_data)
    X_test, y_test = preprocessing(test_df)
    print("Number of windows before thresholds on test : ", len(y_test))
    X_test, y_test, _ = filter_by_thresholding(X=X_test, y=y_test, threshold=threshold)
    print("Number of windows after thresholds on test : ", len(y_test))
    y_pred = model.predict(X_test)

    predictions[f"pred_{test_patient}"] = y_pred
    predictions[f"test_{test_patient}"] = y_test


file_name = "NN_threshold_25"
path = "/sps/crnl/ademasson/data/NN"
save_obj(predictions, file_name, path)
