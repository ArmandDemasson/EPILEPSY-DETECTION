import pickle
import numpy as np
import os
import random
import os.path as op
import string
import sys
import csv
from sklearn.model_selection import KFold
import tensorflow as tf
import classif_models as cm

##################################################################### SET SOME PARAMETERS
seed = 0
tf.keras.utils.set_random_seed(
    seed
)

model_type = "sfcn"
output_path = '/mnt/data/ademasson/CNN/' ############################TO SET
prefix = "data_raw_"

path_extracted_data = '/sps/crnl/pmouches/data/MEG_PHRC_2006_preprocessedNEWAfterFeedback/' ############################TO SET
#subjects = [sub[:-11] for sub in os.listdir(path_extracted_data) if (os.path.isfile(os.path.join(path_extracted_data, sub)) and ('b3_windows' in sub))]
#subjects = sorted(subjects)

subjects = [sub[9:12] for sub in os.listdir(path_extracted_data) if (os.path.isfile(os.path.join(path_extracted_data, sub)) and ('b3_windows' in sub))]

sfreq = 150  # sampling frequency of the data in Hz
window_size = 0.2

training_option = "cv" # can be "ft" if want to fine tune OR "cv" for cross validation OR "auto" for regular training
batch_size = 32
nb_epochs = 1

#For CV only:
nb_cv_splits=len(subjects)
nb_validation_samples=len(subjects)//5
##################################################################### DEFINE FUNCTIONS


def save_obj(obj, name, path):
    # saves an object as pickle file
    with open(op.join(path, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, path):
    # saves a pickle file
    with open(op.join(path, name), 'rb') as f:
        return pickle.load(f)


def generate_database_with_hold_out_memeff(Y,data_all,data_valid,data_test,data_train,balanced=1):  
    # Y: list of labels
    # data_all,data_valid,data_test,data_train: list of all/validation/testint/training subjects
    # balanced: if True generate a simulated balanced dataset. We can balance train and/or test and/or validation datasets.
    # returns the id arrays for each set (train/test/valid)
    
    ############# This functions filling the "ids" array which contains a line for each window with: ids[N,0]=id of the window (in subject space), ids[N,1]=id of the subject, ids[N,2]=label  

    total_nb_window=0
    nb_sub = len(Y)
    for s in range(nb_sub):
        # get the total number of windows to instantiate the "ids" array with the correct shape
        total_nb_window = total_nb_window + len(Y[s])
    ids = np.zeros((total_nb_window ,3),dtype=int)

    start = 0
    stop = 0
    for i in range(nb_sub):
        #for each subject, get the number of windows
        nb_windows = len(Y[i])
        stop = stop + nb_windows
        win_id = np.expand_dims(np.linspace(0,nb_windows-1,num=nb_windows,dtype=int),axis=-1)
        sub_id = np.expand_dims(np.ones((nb_windows),dtype=int)*int(data_all[i]),axis=-1) #store subject id
        # fill the "ids" array with the subject window ids (going from 0 to nb_windows), the subject id for each window and the corresponding label
        ids[start:stop,:]=np.concatenate((win_id,sub_id,np.expand_dims(Y[i][:nb_windows],axis=-1)),axis=-1)
        start = start + nb_windows

    # split the "ids" array into a train, a valid and a test array according to subject ids specified in the function args
    mask=np.isin(ids,data_test)
    X_test_ids = ids[mask[:,1]==True]
    mask=np.isin(ids,data_valid)
    X_valid_ids = ids[mask[:,1]==True]
    mask=np.isin(ids,data_train)
    X_train_ids = ids[mask[:,1]==True]

    print(X_train_ids.shape)
    np.random.shuffle(X_train_ids)
    np.random.shuffle(X_valid_ids)

    if balanced:
        # Comment a block if you don't want to balance all sets
        #Balancing test set 
        np.random.shuffle(X_test_ids)
        X_test_ids = X_test_ids[X_test_ids[:, 2].argsort()[::-1]]
        nb_pos = X_test_ids[X_test_ids[:,2]==1].shape[0]
        X_test_ids = X_test_ids[:2*nb_pos,:]
        np.random.shuffle(X_test_ids)

        #Balancing validation set
        np.random.shuffle(X_valid_ids)
        X_valid_ids = X_valid_ids[X_valid_ids[:, 2].argsort()[::-1]]
        nb_pos = X_valid_ids[X_valid_ids[:,2]==1].shape[0]
        nb_neg = X_valid_ids[X_valid_ids[:,2]==0].shape[0]
        X_valid_ids = X_valid_ids[:2*nb_pos,:]
        np.random.shuffle(X_valid_ids)

        #Balancing training set
        np.random.shuffle(X_train_ids)
        X_train_ids = X_train_ids[X_train_ids[:, 2].argsort()[::-1]]
        nb_pos = X_train_ids[X_train_ids[:,2]==1].shape[0]
        nb_neg = X_train_ids[X_train_ids[:,2]==0].shape[0]
        X_train_ids = X_train_ids[:2*nb_pos,:]
        np.random.shuffle(X_train_ids)
    
    return X_train_ids, X_test_ids, X_valid_ids


def train_model(model_type, path_extracted_data, output_path, sfreq, window_size, X_train_ids, X_test_ids, X_valid_ids, batch_size, nb_epochs, ft=0):
    # model_type: name of the model (for saving files purposes)
    # path_extracted_data: where the .pkl and binary files are
    # output_path: where to save the model file and results
    # sfreq, window_size: params used to create the windows
    # X_train_ids, X_valid_ids, X_test_ids: id arrays obtained from the function generate_database_with_hold_out_memeff
    # tf: if want to fine-tune, contains the path to the initial model weights

    aug = True

    #stuff for dealing with imbalanced data
    # compute class weight to be used in the loss function
    # compute initial bias to be used to initialize the model decision layer weights 
    neg = np.bincount(X_train_ids[:,2])[0]
    pos = np.bincount(X_train_ids[:,2])[1]
    total = X_train_ids.shape[0]
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    initial_bias = np.log([pos/neg])
    print("class_weight: ",class_weight)
    print("initial_bias: ",initial_bias)

    # create the data generators
    training_generator_time, valid_generator_time, testing_generator_time  = cm.load_generators_memeff(X_train_ids, X_valid_ids, X_test_ids, aug, window_size, sfreq, path_extracted_data, batch_size)
    
    # instantiate the model
    model_time, callbacks= eval(f'cm.classify_{model_type}')('time',sfreq,window_size,initial_bias)
    # if fine-tuning based on previous model params, then load the previous model params
    if ft:
        model_time.load_weights(ft,by_name=True)
    # fit the model
    cm.fit_model(model_time, output_path, training_generator_time, valid_generator_time, callbacks, class_weight, batch_size, nb_epochs)        
    # test the model
    y_pred_time = cm.test_model(output_path, model_type, testing_generator_time, X_test_ids, path_extracted_data)


#####################################################################TRAIN/TEST THE MODEL
if training_option == "auto" or training_option == "ft":

    Y=list()
    data_all=list()

    # read training/validation/testing data ids from pre-existing files

    with open(path_extracted_data+'ALlists48/UseAll/training_data.txt', 'r') as f:
        data_train = [line.strip().split('_')[2] for line in f]

    with open(path_extracted_data+'ALlists48/it2/testing_data22_AL.txt', 'r') as f:
        data_test = [line.strip().split('_')[2] for line in f]

    with open(path_extracted_data+'ALlists48/UseAll/holdout.txt', 'r') as f:
        data_valid = [line.strip().split('_')[2] for line in f]

    print(sorted(data_train+data_valid+data_test))

    for ind, sub in enumerate(sorted(data_train+data_valid+data_test)):
        if os.path.isfile(path_extracted_data+"/"+prefix+sub+'_new_labels.pkl'):
            data = load_obj(prefix+sub+'_b3_new_labels.pkl', path_extracted_data)
        else:
            data = load_obj(prefix+sub+'_b3_labels.pkl', path_extracted_data)
        Y.append(data)
        data_all.append(sub)

    data_all = list(map(int, data_all))
    data_valid = list(map(int, data_valid))
    data_train = list(map(int, data_train))
    data_test = list(map(int, data_test))

    print("data_all: ",data_all)
    print("data_train: ",data_train)
    print("data_test: ",data_test)
    print("data_valid: ",data_valid)
    print("labels: ",len(Y))


    print("window labels loaded")

    # Generate "ids" arrays
    X_train_ids, X_test_ids, X_valid_ids = generate_database_with_hold_out_memeff(Y,data_all, data_valid, data_test, data_train)
    # Train/test the model
    if training_option == "ft":
        train_model(model_type, path_extracted_data, output_path, sfreq, window_size, X_train_ids, X_test_ids, X_valid_ids, batch_size, nb_epochs, ft="/mnt/data/pmouches/focalloss/fullytrained2D/testANR/sfcn_time_all_best_model.h5")
    else:
        train_model(model_type, path_extracted_data, output_path, sfreq, window_size, X_train_ids, X_test_ids, X_valid_ids, batch_size, nb_epochs)

elif training_option == "cv":

    # Split CV folds
    kf = KFold(n_splits=nb_cv_splits, random_state=seed, shuffle=True)
    print(subjects)
    print(kf.split(subjects))
    fold=0
    for train, test in kf.split(subjects):

        # For each CV iteration, generate "ids" arrays and train/test the model
        Y=list()
        data_all=list() 

        data_test = [i for i in [subjects[j] for j in test]]
        data_test = list(map(int, data_test))
        train_val_subjects = [subjects[j] for j in train]
        random.shuffle(train_val_subjects)
        data_valid = [i for i in train_val_subjects[:nb_validation_samples]]#keep some subjects for validation
        data_train = [i for i in train_val_subjects[nb_validation_samples:]]#the rest are for training
        data_valid = list(map(int, data_valid))
        data_train = list(map(int, data_train))

        for ind, sub in enumerate(sorted(subjects)):
            if os.path.isfile(path_extracted_data+"/"+prefix+sub+'_b3_new_labels.pkl'):
                data = load_obj(prefix+sub+'_b3_new_labels.pkl', path_extracted_data)
            else:
                data = load_obj(prefix+sub+'_b3_labels.pkl', path_extracted_data)
            blocks = load_obj(prefix+sub+'_b3_blocks.pkl', path_extracted_data)
            Y.append(data)
            data_all.append(sub)

        data_all = list(map(int, data_all))

        print("data_train: ",data_train)
        print("data_test: ",data_test)
        print("data_valid: ",data_valid)
        print("labels: ",len(Y))

        print("window labels loaded")

        X_train_ids, X_test_ids, X_valid_ids = generate_database_with_hold_out_memeff(Y,data_all, data_valid, data_test, data_train)
        
        train_model(model_type, path_extracted_data, output_path, sfreq, window_size, X_train_ids, X_test_ids, X_valid_ids, batch_size, nb_epochs)
        fold = fold+1
