import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_sequence, pad_packed_sequence
from utils import *

import argparse
from config import get_Config
args = get_Config()
device = args.device

"""This is for time masking, TS masking, other function types for TA"""

class SingleHeadAttention_TA(nn.Module):

    def __init__(self, d_model, dropout):
        super().__init__()

        self.temperature = (d_model**(0.5))
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.w_qs2 = nn.Linear(d_model, d_model, bias=False)
        self.w_ks2 = nn.Linear(d_model, d_model, bias=False)

        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.attention = ScaledDotProductAttention_TA(self.temperature, dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, inputs_lengths,distance_matrix):

        sz, max_len = q.size(0), q.size(1)

        q2 = q.clone()
        k2 = k.clone()
        residual = q.clone()

        # shape: b x len x d
        q = self.w_qs(q).view(sz, max_len, -1)
        k = self.w_ks(k).view(sz, max_len, -1)
        v = self.w_vs(v).view(sz, max_len, -1)
        q2 = self.w_qs2(q2).view(sz, max_len, -1)
        k2 = self.w_ks2(k2).view(sz, max_len, -1)

        distance_weights = distance_matrix

        q, attn = self.attention(q, k, v,q2,k2, inputs_lengths,distance_weights)

        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn
    
class ScaledDotProductAttention_TA(nn.Module):

    def __init__(self, temperature, attn_dropout):
        super().__init__()
        
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v,q2,k2, inputs_lengths, distance_weights):

        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        

        q2 = torch.matmul(distance_weights, q2)
        k2 = torch.matmul(distance_weights, k2)
        distance_attn = torch.matmul(q2 / self.temperature, k2.transpose(1, 2))

        attn = attn + distance_attn
        attn = global_zero_masking(attn, inputs_lengths)
        attn = torch.tanh(attn)
        attn = global_zero_masking(attn, inputs_lengths)


        final_attn = attn

        output = torch.matmul(final_attn, v)

        return output, final_attn


class TA_MODEL(nn.Module):
    def __init__(self, input_size,output_size, hidden_size, num_layers, sequence_length, dropout, time_param, function_type = 'sig'):
        super(TA_MODEL, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.dropout = dropout     
        self.time_param = time_param
        self.function_type = function_type
        
        self.a = nn.Parameter(torch.clamp(torch.randn(1, device = device), 0.0001), requires_grad=True)
        self.a2 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.a3 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)
        
        self.lstm = BILSTM(self.input_size, self.output_size, self.hidden_size, self.num_layers, self.sequence_length,self.dropout)
        self.tattn = SingleHeadAttention_TA(2*d_model, self.dropout)
        self.fc = MLP(2*self.hidden_size, self.output_size, self.dropout)
    
    def select_function(self, distance_matrix, function_type):
        
        if function_type == 'sig':
            return 1/(1+torch.exp(-(torch.abs(self.a) * distance_matrix + self.b)))
        
        elif function_type == 'linear':
            return torch.abs(self.a) * distance_matrix + self.b   
        
        elif function_type == 'quad':
            return torch.abs(self.a)*distance_matrix**2 + self.a2*distance_matrix + self.b
        
        elif function_type == 'cubic':
            return torch.abs(self.a)*distance_matrix**3 + self.a2*distance_matrix**2 + self.a3*distance_matrix + self.b
        
        elif function_type == 'exp':
            return torch.exp(torch.abs(self.a) * distance_matrix) + self.b
        
    def forward(self, x, x_time_data):
        
        x, x_lengths = x
        x = x.to(device)
        
        distance_matrix = distance_mat(x_time_data)
        distance_matrix = 1/torch.log(distance_matrix + 2.7)
        param = 1/np.log(self.time_param + 2.7)
        
        # time masking
        distance_matrix, mask = time_masking(distance_matrix, param)        
        distance_matrix = global_zero_masking(distance_matrix, x_lengths)

        # function type default sig
        distance_matrix = self.select_function(distance_matrix, self.function_type)
 
        #distance_matrix = time_masking(distance_matrix, x_lengths, self.sequence_length, param)
        distance_matrix = mask*distance_matrix
        distance_matrix = global_masking(distance_matrix, x_lengths)
        distance_matrix = F.softmax(distance_matrix, dim = -1)
        distance_matrix = distance_matrix.masked_fill(torch.isnan(distance_matrix), 0)   
        
        # LSTM layer
        lstm_out,out_lengths = self.lstm(x, x_lengths)

        # att layer
        att_out, attn = self.tattn(lstm_out,lstm_out,lstm_out, out_lengths, distance_matrix)
             
        # fc
        fc_result = self.fc(att_out.view(-1, 2*self.hidden_size))
        
        # output
        fc_result = fc_result.view(-1, self.sequence_length)
        
        return fc_result,att_out,attn
    
    
    
class SingleHeadAttention_FA(nn.Module):

    def __init__(self, d_model, dropout, feature_param):
        super().__init__()

        self.temperature = (d_model**(0.5))
        self.feature_param = feature_param
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.w_qs2 = nn.Linear(d_model, d_model, bias=False)
        self.w_ks2 = nn.Linear(d_model, d_model, bias=False)

        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.FS = FeatureSim(self.temperature,dropout, self.feature_param)
        self.attention = ScaledDotProductAttention_FA(self.temperature, dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, inputs_lengths,x):

        sz, max_len = q.size(0), q.size(1)

        q2 = q.clone()
        k2 = k.clone()
        residual = q.clone()

        # shape: b x len x d
        q = self.w_qs(q).view(sz, max_len, -1)
        k = self.w_ks(k).view(sz, max_len, -1)
        v = self.w_vs(v).view(sz, max_len, -1)
        q2 = self.w_qs2(q2).view(sz, max_len, -1)
        k2 = self.w_ks2(k2).view(sz, max_len, -1)
        feature_attn = self.FS(x, inputs_lengths)

        q, attn = self.attention(q, k, v,q2,k2, inputs_lengths,feature_attn)

        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn
    
class ScaledDotProductAttention_FA(nn.Module):

    def __init__(self, temperature, attn_dropout):
        super().__init__()
        
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v,q2,k2, inputs_lengths, feature_attn):

        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        

        q2 = torch.matmul(feature_attn, q2)
        k2 = torch.matmul(feature_attn, k2)
        feature_attn = torch.matmul(q2 / self.temperature, k2.transpose(1, 2))

        attn = attn + feature_attn
        attn = global_zero_masking(attn, inputs_lengths)
        attn = torch.tanh(attn)
        attn = global_zero_masking(attn, inputs_lengths)


        final_attn = attn

        output = torch.matmul(final_attn, v)

        return output, final_attn
    
class FeatureSim(nn.Module):
    def __init__(self, temperature, dropout, feature_param):
        super(FeatureSim, self).__init__()
        
        self.temperature = temperature
        self.feature_importance =  nn.Parameter(torch.zeros(11, device = device), requires_grad=True)
        self.dropout = nn.Dropout(dropout)
        self.feature_param = feature_param
        
        
    def forward(self, x, x_lengths):
        
        # eliminate spec feature
        x_feature = x[:,:,:11]
        
        # abs L1 distance
        feature_distance = torch.abs(x_feature.unsqueeze(2)-x_feature.unsqueeze(1))
        
        # weighted sum with feature importance
        feature_distance = feature_distance * self.feature_importance
        feature_attn = torch.sum(feature_distance, dim = -1)
        
        #feature_attn, mask = feature_masking(feature_attn, self.feature_param)
        
        # N% masking
        feature_attn, mask = feature_masking2(feature_attn, self.feature_param)
        feature_attn = feature_attn*mask
        
        feature_attn = global_masking(feature_attn, x_lengths)
        feature_attn = (F.softmax(feature_attn, dim = -1))
        feature_attn = feature_attn.masked_fill(torch.isnan(feature_attn), 0)  

        return feature_attn

class FA_MODEL(nn.Module):
    def __init__(self, input_size,output_size, hidden_size, num_layers, sequence_length, dropout,feature_param):
        super(FA_MODEL, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.feature_param = feature_param
        self.temperature = (hidden_size*2)**0.5

        self.lstm1 = BILSTM(self.input_size, self.output_size, self.hidden_size, self.num_layers, self.sequence_length,dropout = self.dropout).double().to(device)
        self.att = SingleHeadAttention_FA(2*self.hidden_size, self.dropout, self.feature_param).double().to(device)
        self.fc = MLP(2*self.hidden_size, self.output_size, self.dropout).double().to(device)
        

    def forward(self, x, x_time_data):
        
        x, x_lengths = x
        x = x.to(device)

        # LSTM layer
        lstm_out, out_lengths = self.lstm1(x, x_lengths)
        
        #att layer
        att_out, attn = self.att(lstm_out, lstm_out, lstm_out, out_lengths, x)
        
        # fc
        fc_result = att_out.view(-1, self.hidden_size*2)
        fc_result = self.fc(fc_result)
        
        # output
        fc_result = fc_result.view(-1, self.sequence_length)
        
        return fc_result,att_out, attn