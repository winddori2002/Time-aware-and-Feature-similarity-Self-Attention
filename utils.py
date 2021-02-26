import pandas as pd
import random
import numpy as np
from sklearn.utils import shuffle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_sequence, pad_packed_sequence
import neptune

import argparse
from config import get_Config
args = get_Config()
device = args.device

def distance_mat(data):
    # b x n -> b x n x n
    # symmetric time distance matrix
    b,seq = data.size()
    
    distance = torch.cumsum(data[0], dim = 0)      
    distance_matrix = torch.cdist(distance.view(-1,1), distance.view(-1,1))
    batch_matrix = distance_matrix
    for i in range(1, b):
    
        distance = torch.cumsum(data[i], dim = 0)      
        distance_matrix = torch.cdist(distance.view(-1,1), distance.view(-1,1))
        batch_matrix = torch.cat([batch_matrix, distance_matrix])
    batch_matrix = batch_matrix.view(b, seq, seq) 
    return batch_matrix.double()


def neptune_load(experiment_name, tag_name, PARAMS):
    # logging
    neptune.init('winddori/TimePaper2', api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMmU3YzQzNTgtMmVmNC00MmU1LTkzYWUtN2NiNWU5MmE5YTdjIn0=')
    neptune.create_experiment(name=experiment_name, params=PARAMS)
    neptune.append_tag(tag_name)
    
def global_masking(matrix,inputs_lengths):
    
    mask = torch.ones(matrix.size(), requires_grad=False).to(device)
    for i, l in enumerate(inputs_lengths):  # skip the first sentence
        if l < 50:
            mask[i, l:,:] = 0
            mask[i,:,l:] = 0
    
    matrix2 = mask*matrix
    matrix2 = matrix2.masked_fill(mask == 0, float('-inf'))
    return matrix2

def global_zero_masking(matrix,inputs_lengths):
    
    mask = torch.ones(matrix.size(), requires_grad=False).to(device)
    for i, l in enumerate(inputs_lengths):  # skip the first sentence
        if l < 50:
            mask[i, l:,:] = 0
            mask[i,:,l:] = 0
    
    matrix2 = mask*matrix
    return matrix2

def pad_masking(matrix,inputs_lengths):
    
    mask = torch.ones(matrix.size(), requires_grad=False).double().to(device)
    for i, l in enumerate(inputs_lengths):  # skip the first sentence
        if l < 50:
            mask[i,:,l:,:] = 0
            mask[i,:,:,l:] = 0
    
    matrix2 = mask*matrix
    #matrix2 = matrix2.masked_fill(mask == 0, 1e-15)
    matrix2 = matrix2.masked_fill(mask == 0, float('-inf'))
    return matrix2

def cosine_similarity(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def time_masking2(matrix,inputs_lengths, sequence_lengths, time_param):
    
    mask = torch.ones(matrix.size(), requires_grad=False).to(device)
    for i, l in enumerate(inputs_lengths):  # skip the first sentence
        if l < sequence_lengths:
            mask[i, l:,:] = 0
            mask[i,:,l:] = 0  
            
    matrix2 = mask*matrix   
    mask2 = matrix2 > time_param
    matrix3 = mask2*matrix2
    
    matrix3 = matrix3.masked_fill(mask == 0, float('-inf'))
    return matrix3

def time_masking(matrix, time_param):
    
    mask = matrix > time_param
    matrix2 = mask*matrix
    
    return matrix, mask

def feature_masking(matrix, feature_param):
    
    
    mask = matrix > feature_param
    matrix2 = mask*matrix
    
    return matrix, mask

def feature_masking2(matrix, feature_param):
    
    matrix2 = matrix.clone()
    matrix2 = torch.argsort(matrix2, dim = -1)
    
    mask = torch.where(matrix2<feature_param, torch.ones(1, device=device), torch.zeros(1,device=device))
    
    return matrix, mask

def model_connection(f_model, t_model, x_data, time_data, out_type):
    
    f_model.eval()
    t_model.eval()
    
    with torch.no_grad():
        
        f_outputs = f_model(x_data, time_data)
        t_outputs = t_model(x_data, time_data)
    
    if out_type == 'output':
        f_outputs = f_outputs[0]
        t_outputs = t_outputs[0]
    elif out_type == 'representation':
        f_outputs = f_outputs[1]
        t_outputs = t_outputs[1]        
        
    return f_outputs, t_outputs


def sequential_sorting(raw_data, sequence_len):
    ship = raw_data.copy()
    ship = ship.sort_values(['VESSELCODE','UTC'])
    ship.index = list(range(len(ship)))    
    
    ship['UTC_LAG'] = ship['UTC'].shift(1)
    ship['UTC_DIFF'] = ship['UTC'] - ship['UTC_LAG']
    ship['UTC_DIFF'] = ship['UTC_DIFF'].dt.total_seconds()/60
    
    ship['UTC_DIFF'].loc[0] = 0    
    a = 0
    a_list = []
    
    for i in range(len(ship)):
        if i == 0:
            a_list.append(a)
            continue
            
        elif (ship['VESSELCODE'][i] != ship['VESSELCODE'][i-1]):
            a = 0
            a_list.append(a)
        
        else:
            a += 1
            a_list.append(a)       
            
    ship['SEQUENCE'] = a_list     
    seq_idx = ship[ship['SEQUENCE']==0].index
    ship['UTC_DIFF'].loc[seq_idx] = 0
    ship['UTC_DIFF'] = np.round(ship['UTC_DIFF'])
    
    a = 0
    a_list = []
    
    for i in range(len(ship)):
        if i == 0:
            a_list.append(a)
            a+=1
            continue
        
        elif ship['SEQUENCE'][i] == 0:
            a = 0
            a_list.append(a)
            a+=1    
            
        elif ship['UTC_DIFF'][i] > 60*6:
            a = 0
            a_list.append(a)
            a+=1
            
        else:
            a_list.append(a)
            a+=1
            if a == sequence_len:
                a = 0
            
    ship['SEQUENCE2'] = a_list
    seq_idx = ship[ship['SEQUENCE2']==0].index
    ship['UTC_DIFF'].loc[seq_idx] = 0
    
    return ship


def select_setting(args):
    if args.att_type == 'ENS':
        model_setting = {'input_size': args.input_size,
                         'output_size': args.output_size,
                         'hidden_size': args.hidden_size,
                         'sequence_length': args.sequence_length,
                         'num_layers': args.num_layers,
                         'dropout': args.dropout}  

        model_setting_ens = {'input_size': args.input_size,
                         'output_size': args.output_size,
                         'hidden_size': args.hidden_size,
                         'sequence_length': args.sequence_length,
                         'num_layers': args.num_layers,
                         'dropout': args.dropout,    
                         'out_type': args.out_type}  
        return model_setting_ens

    else:
        model_setting = {'input_size': args.input_size,
                         'output_size': args.output_size,
                         'hidden_size': args.hidden_size,
                         'sequence_length': args.sequence_length,
                         'num_layers': args.num_layers,
                         'dropout': args.dropout}          
        return model_setting
