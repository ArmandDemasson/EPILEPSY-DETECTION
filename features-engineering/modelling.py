import numpy as np
import pandas as pd
import os.path as op
from sklearn.feature_selection import SelectKBest, chi2
import pickle
from scipy.fft import fft, fftfreq
from tqdm import tqdm
from sklearn.model_selection import KFold

def load_obj(name, path):
    with open(op.join(path, name), 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name, path):
    with open(op.join(path, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

path = "/sps/crnl/ademasson/data/features_extraction"

balanced_patients_df = load_obj("balanced_preprocessed_data.pkl",path)
patients_df = load_obj("preprocessed_data.pkl",path)
patients = ["25", "27", "38", "61", "83", "97", "18", "22", "54", "68", "69", "84", "105", "42", "20", "30", "11", "47", "56", "101", "117", "4", "8", "26", "66", "67", "79", "92", "111", "2", "24", "65", "73", "100", "118", "119", "23", "35","40", "62", "102", "107", "108", "109"]

nb_cv_splits=len(patients)
kf = KFold(n_splits=nb_cv_splits, random_state=0, shuffle=True)
cv_set = []
for train_patients, test_patients in kf.splits(patients):
    print(test_patients)
    print(train_patients)
    # Train set
    X_train = None
    y_train = None
    for j, patient in enumerate(train_patients):
        X_patient_set = balanced_patients_df[f"X_{patient}"]
        y_patient_set = balanced_patients_df[f"y_{patient}"]
        print(X_patient_set.shape)
        print(y_patient_set.shape)
        if X_train is None:
            X_train = X_patient_set
        else:
            X_train = pd.concat([X_train, X_patient_set], axis=0, ignore_index=True)

        if y_train is None:
            y_train = y_patient_set
        else:
            y_train = np.hstack([y_train, y_patient_set])
            
    # Test sets
    X_test = None
    y_test = None
    for j, patient in enumerate(test_patients):
        X_patient_set = patients_df[f"X_{patient}"]
        y_patient_set = patients_df[f"y_{patient}"]             
        if X_test is None:
            X_test = X_patient_set
        else:
            X_test = pd.concat([X_test, X_patient_set], axis=0, ignore_index=True)

        if y_test is None:
            y_test = y_patient_set
        else:
            y_test = np.hstack([y_test, y_patient_set])
    cv_set.append((X_train, y_train, X_test, y_test))

features_columns = {}

list_of_features = ["std", "ppa", "average_slope", "upslope", "amplitude_ratio", "sharpness", "down_slope", "main_frequency", "phase_congruency"]

for feature in list_of_features:
    matching_columns = [col for col in X_train.columns if col.startswith(feature)]
    features_columns[feature] = matching_columns
    

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers

tf.random.set_seed(0)

predictions = {}
for i, cv_ in enumerate(cv_set):
    X_train, y_train, X_test, y_test = cv_
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    model = keras.Sequential(
        [
            keras.Input(shape=(X_train.shape[-1],)),
            layers.Dense(20, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss=keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"])

    history=model.fit(X_train, y_train, epochs=50, validation_split = 0.2, batch_size=32)
    y_pred = model.predict(X_test)
    y_pred = [0 if y < 0.95 else 1 for y in y_pred]
    predictions[f"agr_test_cv_{i}"] = y_test
    predictions[f"agr_pred_cv_{i}"] = y_pred


file_name = "modelling_balanced_agregated_feature"
save_obj(predictions, file_name, path)
