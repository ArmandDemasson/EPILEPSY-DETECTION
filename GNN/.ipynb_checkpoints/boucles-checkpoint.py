import torch
#from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
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
import params

def weighted_binary_cross_entropy(output, target, weights=None, reduction="mean"):
        output = torch.clamp(output,min=1e-7,max=1-1e-7)
        if weights is not None:           
            loss = weights[1] * (target * torch.log(output)) + weights[0] * ((1 - target) * torch.log(1 - output))
        else:
            loss = (target * torch.log(output)) + ((1 - target) * torch.log(1 - output))
        if reduction=="mean":
            return torch.neg(torch.mean(loss))
        else:
            return torch.neg(loss)  

def sigmoid_focal_loss(inputs,targets,alpha = params.alpha_fl ,gamma = params.gamma_fl,reduction ="mean", weights=None):
    
    p = inputs
    ce_loss = weighted_binary_cross_entropy(inputs, targets, weights=weights, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss

def compute_features(batch, device):
    features = torch.empty(len(batch.to_data_list()),274,4)
    i=0
    for graph in batch.to_data_list() :
        features[i]=graph.x
        i+=1
    features = features.view(len(batch.to_data_list()),1,274,4)
    features = features.to(device)
    inputs, edge_index, edge_attr, labels= batch.x, batch.edge_index, batch.edge_attr, batch.y
    inputs = inputs.to(device)  
    labels = labels.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    edge_attr = edge_attr.float() 

    batch = batch.to(device)  

    inputs = inputs.float()
    

    return features, batch


def train_epoch(model, dataloader, optimizer, device, class_weight, need_features=False):
    model.train()
    loss_train = []
    criterion = nn.BCELoss(reduction='sum')
    for data in dataloader:

        if need_features:
            features, batch = compute_features(data, device)
            labels = batch.y
            outputs = model(features, batch)
        else:
            inputs, labels = data
            inputs = inputs.to(device)
            inputs = inputs.float()
            outputs = model(inputs)

        labels = labels.to(device)
        labels = labels.float()
        labels = labels.squeeze()
        outputs = outputs.squeeze()

        try : 
            loss = criterion(outputs, labels)
#             loss = sigmoid_focal_loss(outputs, labels,reduction='sum')
            loss_train.append(loss.item())
        except RuntimeError as e:
            print(f"Skipping batch due to error: {e}")
            continue
        except IndexError as i : 
            print(f"Skipping batch due to error: {i}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return mean(loss_train)


def validate_epoch(model, dataloader, device, class_weight, need_features=False):
    model.eval()
    loss_valid = []

    with torch.no_grad():
        for data  in dataloader:

            if need_features:
                features, batch = compute_features(data,device)
                labels = batch.y
                outputs = model(features, batch)
            else:
                inputs, labels = data
                inputs = inputs.to(device)
                inputs = inputs.float()
                outputs, _ = model(inputs)

            labels = labels.to(device)
            labels = labels.float()
            labels = labels.squeeze()
            outputs = outputs.squeeze()

            validation_loss = sigmoid_focal_loss(outputs, labels,reduction='sum')
            #print(validation_loss)
                
            loss_valid.append(validation_loss.item())
    return mean(loss_valid)


def test_model(model, dataloader, device, need_features=False):
    model.eval() 

    detected_spike_list_raw=np.empty((0))
    detected_spike_list_thresh=np.empty((0))
    true_spike_list=np.empty((0))

    reading_embeddings_cnn = np.empty((0))
    reading_labels = np.empty((0))
    reading_outputs = np.empty((0))

    with torch.no_grad():
        for data  in dataloader:

            if need_features:
                features, batch = compute_features(data,device)
                labels = batch.y
                outputs = model(features, batch)
            else:
                inputs, labels = data
                inputs = inputs.to(device)
                inputs = inputs.float()
                outputs = model(inputs)

            labels = labels.to(device)
            labels = labels.float()
            labels = labels.squeeze()
            
            predicted = ((outputs) > 0.5).int()
            predicted = torch.squeeze(predicted)
            

            detected_spike_list_raw = np.append(detected_spike_list_raw,outputs.cpu().numpy())
            detected_spike_list_thresh = np.append(detected_spike_list_thresh,predicted.cpu().numpy())
            true_spike_list = np.append(true_spike_list,labels.cpu().numpy())

            """reading_embeddings_cnn = np.append(reading_embeddings_cnn, emb.cpu().numpy())
            reading_labels = np.append(reading_labels, labels.cpu().numpy())
            reading_outputs = np.append(reading_outputs, outputs.cpu().numpy())"""
    
    return true_spike_list, detected_spike_list_raw, detected_spike_list_thresh, reading_embeddings_cnn, reading_labels, reading_outputs 
