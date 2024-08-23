import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
import gc
import pickle
import random
from warnings import warn
import string
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
from scipy import stats
from sklearn.model_selection import ParameterGrid
import time
from statistics import mean
from torch.optim.lr_scheduler import StepLR, LambdaLR

# Local imports
from boucles import validate_epoch, test_model, train_epoch
from model import EEGGraphChebConvNet
import saving_functions as save
import params
from Hybrid_dataset import EEGGraphDataset

if params.need_features:
  from torch_geometric.loader import DataLoader
  from torch_geometric.data import Batch, Data
else:
  from torch.utils.data import Dataset, DataLoader

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    model.load_state_dict(torch.load(filename))

def save_obj(obj, name, path):
  with open(path+ name + '.pkl', 'wb') as f:
      pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, path):
  with open(path+ name, 'rb') as f:
      return pickle.load(f)

def z_score_normalize(data):
    return (data - data.mean()) / data.std()

def generate_database_with_hold_out_memeff(Y,data_all,data_valid,data_test,data_train,rs,balanced=0,):  
  # index_hold_out: list of ids to hold_out
  # data_valid: list of validation subject ids
  # data_train: list of training subject ids
  # data_test: list of testing subject ids
  # data_all: list of all subject ids, in the same order as in Y
  # Y: (list of list of window labels, in the same order as in data_all. (Y[0] contains window labels of suject data_all[0])
  # balanced: If want to undersample non-spike windows to generated a balanced database

  total_nb_window=0
  nb_sub = len(Y)
  # Go through all subjects and add their number of windows to "total_nb_window"
  for s in range(nb_sub):
      total_nb_window = total_nb_window + len(Y[s])

  # filling "ids" array of shape(total_nb_windows,3)
  # For each window, ids[window]=[window_id, subject_id, label]
  # This big array is then used to go read the correct infos in windows_bi file during model training process
  ids = np.zeros((total_nb_window ,3),dtype=int)
  # "start" and "stop" index used to fill the correct part of "ids" when looping through the subjects
  start = 0
  stop = 0
  for i in range(nb_sub):
      nb_windows = len(Y[i])
      stop = stop + nb_windows
      # For each subject, generate window ids (going from 0 to nb_windows of subject i)
      win_id = np.expand_dims(np.linspace(0,nb_windows-1,num=nb_windows,dtype=int),axis=-1)
      # Generate a numpy array with repeated subject i id 
      sub_id = np.expand_dims(np.ones((nb_windows),dtype=int)*int(data_all[i]),axis=-1)
      # Fill "ids" array with [window_id, subject_id, label] for subject i
      ids[start:stop,:]=np.concatenate((win_id,sub_id,np.expand_dims(Y[i][:nb_windows],axis=-1)),axis=-1)
      start = start + nb_windows

  # From the "ids" array, extract valid, test and train subjects based on data_valid,data_test,data_train
  mask=np.isin(ids,data_test)
  X_test_ids = ids[mask[:,1]==True]
  mask=np.isin(ids,data_valid)
  X_valid_ids = ids[mask[:,1]==True]
  mask=np.isin(ids,data_train)
  X_train_ids = ids[mask[:,1]==True]

  #print("X_train_shape :",X_train_ids.shape)
  np.random.seed(rs)
  np.random.shuffle(X_train_ids)
  np.random.seed(rs)
  np.random.shuffle(X_valid_ids)

  X_test_ids_balanced = X_test_ids.copy()

  if balanced:
      # undersample X_train_ids to get as many non-spike windows as spike windows
    X_train_ids = X_train_ids[X_train_ids[:, 2].argsort()[::-1]]
    nb_pos = X_train_ids[X_train_ids[:,2]==1].shape[0]
    nb_neg = X_train_ids[X_train_ids[:,2]==0].shape[0]
    X_train_ids = X_train_ids[:2*nb_pos,:]
    np.random.seed(rs)
    np.random.shuffle(X_train_ids)

    np.random.seed(rs)
    np.random.shuffle(X_test_ids_balanced)
    X_test_ids_balanced = X_test_ids_balanced[X_test_ids_balanced[:, 2].argsort()[::-1]]
    nb_pos = X_test_ids_balanced[X_test_ids_balanced[:,2]==1].shape[0]
    nb_neg = X_test_ids_balanced[X_test_ids_balanced[:,2]==0].shape[0]
    X_test_ids_balanced = X_test_ids_balanced[:2*nb_pos,:]
    np.random.seed(rs)
    np.random.shuffle(X_test_ids_balanced)

    X_valid_ids = X_valid_ids[X_valid_ids[:, 2].argsort()[::-1]]
    nb_pos = X_valid_ids[X_valid_ids[:,2]==1].shape[0]
    nb_neg = X_valid_ids[X_valid_ids[:,2]==0].shape[0]
    X_valid_ids = X_valid_ids[:2*nb_pos,:]
    np.random.seed(rs)
    np.random.shuffle(X_valid_ids)
  
  return X_train_ids, X_test_ids, X_test_ids_balanced, X_valid_ids #[item for sublist in Y_test for item in sublist]

def get_initial_bias(X_train_ids):

  #stuff for dealing with imbalanced data
  # Get training data balance to initialize the bias weight of the decision layer (last layer) of the CNN
  neg = np.bincount(X_train_ids[:,2])[0]
  pos = np.bincount(X_train_ids[:,2])[1]
  total = X_train_ids.shape[0]
  weight_for_0 = (1 / neg) * (total / 2.0)
  weight_for_1 = (1 / pos) * (total / 2.0)
  class_weight = [weight_for_0, weight_for_1]
  #print(class_weight)
  initial_bias = np.log([pos/neg])
  return initial_bias,class_weight

#########################Get subject list##########################
subjects = sorted(params.subjects)

##############################Some parameters##############################

##############################CV##############################


def model_train_test(X_train_ids,X_test_ids,X_test_ids_balanced,train_dataloader,valid_dataloader,test_dataloader,test_window_dataset_balanced,fold):
    initial_bias,class_weight = get_initial_bias(X_train_ids)
    model = EEGGraphChebConvNet()
    model_name = 'ChebConv_corr_90%'

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    pars = sum([np.prod(p.size()) for p in model_parameters])
    print("This model has nb parameters: ", pars)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate,weight_decay=params.weight_decay)  

    def lr_lambda(epoch):
        if epoch < params.warmup_epochs:
          return epoch / params.warmup_epochs
        else:
          return params.beta ** ((epoch-params.warmup_epochs) // params.step_size)
    
    #scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Accès au GPU
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)
    
    # Boucle d'entraînement
    epochs_no_improve = 0
    j=0
    loss_valid_all =[]
    loss_train_all =[]

    for epoch in range(params.num_epochs):

        training_loss = train_epoch(model, train_dataloader, optimizer, device, class_weight, need_features=params.need_features)

        loss_train_all.append(training_loss)
        #scheduler.step()
        
        best_val_loss = 100000
        loss_valid = []
        true_labels = []
        pred_scores = []    

        validation_loss = validate_epoch(model, valid_dataloader, device, class_weight, need_features=params.need_features)
        if validation_loss < best_val_loss:
            # checkpoint(model, path_writing_data_repeated+'best_ChebConv_'+str(fold)+'.pth')
            best_val_loss = validation_loss

        loss_valid_all.append(validation_loss)

    print("Train loss : ",loss_train_all)
    print("Valid loss :",loss_valid_all )

#     save.plot_epochs_metric(loss_train_all, loss_valid_all, path_writing_data_repeated, model_name, fold)

    print("Finished Training")

    # resume(model, path_writing_data_repeated+'best_ChebConv_'+str(fold)+'.pth')

    true_spike_list, detected_spike_list_raw, detected_spike_list_thresh, reading_embeddings_cnn, reading_labels, reading_outputs = test_model(model, test_dataloader, device, need_features=params.need_features)
        
    save.save_model_results(path_writing_data_repeated,model_name,true_spike_list, detected_spike_list_thresh,fold)
    save.save_model_predictions(X_test_ids,params.path_extracted_data,path_writing_data_repeated,model_name,true_spike_list,detected_spike_list_raw)
    
    if params.save_emb:
        save.save_model_embeddings(reading_embeddings_cnn, reading_labels, reading_outputs,path_writing_data_repeated,model_name,fold)
    true_spike_list, detected_spike_list_raw, detected_spike_list_thresh, reading_embeddings_cnn, reading_labels, reading_outputs = test_model(model, test_dataloader_balanced, device, need_features=params.need_features)

    save.save_model_results(path_writing_data_repeated,model_name+"_balanced_test",true_spike_list, detected_spike_list_thresh,fold)
    save.save_model_predictions(X_test_ids_balanced,params.path_extracted_data,path_writing_data_repeated,model_name+"_balanced_test",true_spike_list,detected_spike_list_raw)
    if params.save_emb:
        save.save_model_embeddings(reading_embeddings_cnn, reading_labels, reading_outputs,path_writing_data_repeated,model_name+"_balanced_test",fold)

torch.manual_seed(43)#42)

if sys.argv[1]=="cv":
    rs=0
    kf = KFold(n_splits = params.splits,shuffle = True,random_state=rs)#1
    fold=0

    path_writing_data_repeated = params.path_writing_data
    
    print("###################################################")
    print(params.model)
    print(params.splits)
    print("###################################################")

    for train, test in kf.split(subjects):
        fold = fold+1
        print("Fold :",  fold)
        
        Y=list()
        data_all=list()
        
        Y_add = list()
        data_add_spikes = list()
        
        train_val_subjects = [subjects[j] for j in train]
        test_subject = [subjects[j] for j in test]
        random.seed(2)
        random.shuffle(train_val_subjects)
        
        # first 5 subjects for training
        data_train = [i for i in train_val_subjects[:-params.nb_valid_subjects]]
        data_train = list(map(int, data_train))
        
        
        # next subject for validation
        data_valid = [i for i in train_val_subjects[-params.nb_valid_subjects:]]
        data_valid = list(map(int, data_valid))
        
        
        # last subject for testing
        data_test = [i for i in test_subject]
        data_test = list(map(int, data_test))
        
        print("Data_train, data_valid, Data test",data_train,data_valid, data_test)
        
        for ind, sub in enumerate(sorted(subjects)):
            data = load_obj('data_raw_'+sub.zfill(3)+'_b3_new_labels.pkl', params.path_extracted_data)
            Y.append(data)
            data_all.append(sub)
        
        data_all = list(map(int, data_all))
        
        X_train_ids, X_test_ids, X_test_ids_balanced, X_valid_ids = generate_database_with_hold_out_memeff(Y,data_all, data_valid, data_test, data_train, rs, balanced=params.balanced)
        
        if params.need_features:
            train_window_dataset = EEGGraphDataset(X_train_ids.tolist(),params.dim,params.path_extracted_data, subjects=data_train)
            train_dataloader = DataLoader(train_window_dataset, batch_size=params.batch_size, shuffle=True)
            
            test_window_dataset = EEGGraphDataset(X_test_ids.tolist(),params.dim,params.path_extracted_data, subjects=data_test)
            test_dataloader = DataLoader(test_window_dataset, batch_size=params.batch_size)
            
            test_window_dataset_balanced = EEGGraphDataset(X_test_ids_balanced.tolist(),params.dim,params.path_extracted_data, subjects=data_test)
            test_dataloader_balanced = DataLoader(test_window_dataset_balanced, batch_size=params.batch_size)
            
            valid_window_dataset = EEGGraphDataset(X_valid_ids.tolist(),params.dim,params.path_extracted_data, subjects= data_valid)
            valid_dataloader = DataLoader(valid_window_dataset, batch_size=params.batch_size, shuffle=True)
            
        else: 
            train_window_dataset = TimeSeriesDataSet(X_train_ids.tolist(),params.dim,params.path_extracted_data, aug=params.augmentation)
            test_window_dataset = TimeSeriesDataSet(X_test_ids.tolist(),params.dim,params.path_extracted_data)
            test_window_dataset_balanced = TimeSeriesDataSet(X_test_ids_balanced.tolist(),params.dim,params.path_extracted_data)
            valid_window_dataset = TimeSeriesDataSet(X_valid_ids.tolist(),params.dim,params.path_extracted_data)
        
            train_dataloader = DataLoader(train_window_dataset, batch_size=params.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_window_dataset, batch_size=params.batch_size)
            test_dataloader_balanced = DataLoader(test_window_dataset_balanced, batch_size=params.batch_size)
            valid_dataloader = DataLoader(valid_window_dataset, batch_size=params.batch_size, shuffle=True)
        model_train_test(X_train_ids,X_test_ids,X_test_ids_balanced,train_dataloader,valid_dataloader,test_dataloader,test_window_dataset_balanced,fold)

elif sys.argv[1]=="auto":
    
    fold=100
    
    Y=list()
    data_all=list()
    
    with open(params.path_extracted_data+'ALlists48/it2/training_data21_AL.txt', 'r') as f:
        data_train = [line.strip().split('_')[2] for line in f]

    print(data_train)
    
    with open(params.path_extracted_data+'ALlists48/it2/testing_data21_AL.txt', 'r') as f:#testing_data11_AL.txt#hold_out_data.txt
        data_test = [line.strip().split('_')[2] for line in f]
    
    print(data_test)
    
    with open(params.path_extracted_data+'ALlists48/it2/valid_data2_AL.txt', 'r') as f:
        data_valid = [line.strip().split('_')[2] for line in f]
    
    print(data_valid)
    
    print(sorted(data_train+data_valid+data_test))

    for ind, sub in enumerate(sorted(data_train+data_valid+data_test)):
        data = load_obj('data_raw_'+sub+'_b3_labels.pkl', params.path_extracted_data)
        Y.append(data)
        data_all.append(sub)

    data_all = list(map(int, data_all))
    data_valid = list(map(int, data_valid))
    data_train = list(map(int, data_train))
    data_test = list(map(int, data_test))
    
    X_train_ids, X_test_ids, X_valid_ids = generate_database_with_hold_out_memeff(Y,data_all, data_valid, data_test, data_train)
    
    train_window_dataset = TimeSeriesDataSet(X_train_ids.tolist(),params.dim,params.path_extracted_data, aug=params.augmentation)
    train_dataloader = DataLoader(train_window_dataset, batch_size=params.batch_size, shuffle=True)
    
    test_window_dataset = TimeSeriesDataSet(X_test_ids.tolist(),params.dim,params.path_extracted_data)
    test_dataloader = DataLoader(test_window_dataset, batch_size=params.batch_size, shuffle=False)
    
    valid_window_dataset = TimeSeriesDataSet(X_valid_ids.tolist(),params.dim,params.path_extracted_data)
    valid_dataloader = DataLoader(valid_window_dataset, batch_size=params.batch_size, shuffle=True)
    
    #model_train_test(X_train_ids,train_dataloader,valid_dataloader,test_dataloader,fold)

