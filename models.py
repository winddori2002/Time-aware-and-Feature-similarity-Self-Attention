import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_sequence, pad_packed_sequence

from utils import *
from attention_modules import *

import argparse
from config import get_Config
args = get_Config()
device = args.device

class Sequence_model(nn.Module):
    """
        Chosse base sequence model with model type
        model_type: LSTM / GRU
        bidirection: True / False
    """
    def __init__(self, rnn_input_size, rnn_hidden, rnn_output_size, rnn_num_layers, sequence_length, dropout, bidirection, model_type):
        super().__init__()
        
        if model_type == 'LSTM':
            self.lstm = nn.LSTM(input_size=rnn_input_size,
                                hidden_size=rnn_hidden,
                                batch_first=True, num_layers=rnn_num_layers, dropout=dropout, bidirectional=bidirection)
        else:
            self.lstm = nn.GRU(input_size=rnn_input_size,
                               hidden_size=rnn_hidden,
                               batch_first=True, num_layers=rnn_num_layers, dropout=dropout, bidirectional=bidirection)
        
    def forward(self, x, x_lengths):
        
        packed_input = pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        lstm_out, hidden = self.lstm(packed_input)
        lstm_out, out_lengths = pad_packed_sequence(lstm_out, batch_first=True)
        
        return lstm_out, out_lengths
    

class MLP(nn.Module):
    """
        Simple MLP with three layers
        This is for prediction layer and MLP models
    """   
    def __init__(self,input_size, output_size, dropout):
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        
        self.dropout1 = nn.Dropout(p = self.dropout)
        self.layer1 = nn.Linear(self.input_size , 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(p = self.dropout)
        self.layer2 = nn.Linear(32, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.layer3 = nn.Linear(32, self.output_size)

        
    def forward(self, x):
        x = self.dropout1(x)
        x = F.relu(self.layer1(x))
        x = self.bn1(x)
        x = self.dropout2(x)
        x = F.relu(self.layer2(x))
        x = self.bn2(x)
        x = self.layer3(x)
        
        return x



class BILSTM(nn.Module):
    """
        Backbone for TA, FA, ENS
    """   
    def __init__(self,input_size,output_size, hidden_size, num_layers, sequence_length,dropout):
        super(BILSTM, self).__init__()
        
        self.output_size = output_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, num_layers= self.num_layers,dropout = self.dropout, bidirectional = True)
        self.hidden = None
        
    def init_hidden(self,x):
                                         # layer, batch_size, hidden_size
        return (Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).double().to(device),
                Variable(torch.zeros(2*self.num_layers, x.size(0), self.hidden_size)).double().to(device))
    
    def forward(self, x, x_lengths):
        
        self.hidden = self.init_hidden(x)
        packed_input = pack_padded_sequence(x, x_lengths, batch_first = True, enforce_sorted=False)

        lstm_out, self.hidden = self.lstm(packed_input, self.hidden)
        lstm_out, out_lengths = pad_packed_sequence(lstm_out, batch_first = True)
        
        return lstm_out, out_lengths
    
class BASE_MODEL(nn.Module):
    """
        Sequence model with prediction layer
    """       
    def __init__(self, input_size,output_size, hidden_size, num_layers, sequence_length, dropout):
        super(BASE_MODEL, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.dropout = dropout
        
        self.lstm1 = BILSTM(self.input_size, self.output_size, self.hidden_size, self.num_layers, self.sequence_length,dropout = self.dropout).double().to(device)
        self.fc = MLP(2*self.hidden_size, self.output_size, self.dropout).double().to(device)

    def forward(self, x, x_time_data):
        
        x, x_lengths = x
        
        # LSTM layer 
        lstm_out, out_lengths = self.lstm1(x, x_lengths)
        
        #att layer no attention
        att_out = lstm_out
        
        # fc
        fc_result = att_out.view(-1, self.hidden_size*2)
        fc_result = self.fc(fc_result)
        
        # output
        fc_result = fc_result.view(-1, self.sequence_length)
        
        return fc_result, out_lengths

class SA_MODEL(nn.Module):
    """
        Sequence model with Self attention
    """   
    def __init__(self, input_size,output_size, hidden_size, num_layers, sequence_length, dropout):
        super(SA_MODEL, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.dropout = dropout
        
        self.lstm1 = BILSTM(self.input_size, self.output_size, self.hidden_size, self.num_layers, self.sequence_length,dropout = self.dropout).double().to(device)
        self.att = SingleHeadAttention(2*self.hidden_size, self.dropout).double().to(device)
        self.fc = MLP(2*self.hidden_size, self.output_size, self.dropout).double().to(device)

    def forward(self, x, x_time_data):
        
        x, x_lengths = x
        
        # LSTM layer
        lstm_out, out_lengths = self.lstm1(x, x_lengths)
        
        #att layer
        att_out, attn = self.att(lstm_out, lstm_out, lstm_out, out_lengths)
        
        # fc
        fc_result = att_out.view(-1, self.hidden_size*2)
        fc_result = self.fc(fc_result)
        
        # output
        fc_result = fc_result.view(-1, self.sequence_length)
        
        return fc_result, attn    

class FA_MODEL(nn.Module):
    """
        Sequence model with Feature attention
    """   
    def __init__(self, input_size,output_size, hidden_size, num_layers, sequence_length, dropout):
        super(FA_MODEL, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.temperature = (hidden_size*2)**0.5

        self.lstm1 = BILSTM(self.input_size, self.output_size, self.hidden_size, self.num_layers, self.sequence_length,dropout = self.dropout).double().to(device)
        self.att = SingleHeadAttention_FA(2*self.hidden_size, self.dropout).double().to(device)
        self.fc = MLP(2*self.hidden_size, self.output_size, self.dropout).double().to(device)

    def forward(self, x, x_time_data):
        
        x, x_lengths = x

        # LSTM layer
        lstm_out, out_lengths = self.lstm1(x, x_lengths)
        
        #att layer 
        att_out, attn = self.att(lstm_out, lstm_out, lstm_out, out_lengths, x)
        
        # fc
        fc_result = att_out.view(-1, self.hidden_size*2)
        fc_result = self.fc(fc_result)
        
        # output 
        fc_result = fc_result.view(-1, self.sequence_length)
        
        return fc_result, att_out,attn

    
class TA_MODEL(nn.Module):
    """
        Sequence model with Time attention
    """   
    def __init__(self, input_size,output_size, hidden_size, num_layers, sequence_length, dropout):
        super(TA_MODEL, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.dropout = dropout     
        self.a = nn.Parameter(torch.clamp(torch.randn(1, device = device), 0.0001), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)
        
        self.lstm = BILSTM(self.input_size, self.output_size, self.hidden_size, self.num_layers, self.sequence_length,self.dropout)
        self.tattn = SingleHeadAttention_TA(2*hidden_size, self.dropout)
        self.fc = MLP(2*self.hidden_size, self.output_size, self.dropout)

    def forward(self, x, x_time_data):
        
        x, x_lengths = x
        
        # b x n -> b x n x n 
        # b x n x n time matrix scaling
        distance_matrix = distance_mat(x_time_data)
        distance_matrix = 1/torch.log(distance_matrix + 2.7)
        
        # sigmoid function representation
        distance_matrix = global_zero_masking(distance_matrix, x_lengths)
        distance_matrix = 1/(1+torch.exp(-(torch.abs(self.a) * distance_matrix + self.b)))
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
        
        return fc_result, att_out, attn   
    
    
class Ensemble_layer(nn.Module):
    """
        Ensemble prediciton layer
    """   
    def __init__(self,input_size, output_size, dropout):
        super(Ensemble_layer, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        
        self.dropout1 = nn.Dropout(p = self.dropout)
        self.layer1 = nn.Linear(self.input_size , 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(p = self.dropout)
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.layer3 = nn.Linear(32, self.output_size)
        
    def forward(self, x):
        x = self.dropout1(x)
        x = F.relu(self.layer1(x))
        x = self.bn1(x)
        x = self.dropout2(x)
        x = F.relu(self.layer2(x))
        x = self.bn2(x)
        x = self.layer3(x)
        return x

class Ensemble(nn.Module):
    """
        Ensemble model
        out_type: Select inputs type of TA and FA to ensemble
    """   
    def __init__(self, input_size,output_size, hidden_size, num_layers, sequence_length, dropout, out_type):
        super(Ensemble, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.dropout = dropout     
        if out_type == 'output':
            self.fc_input_size = 2
        elif out_type == 'representation':
            self.fc_input_size = 4*hidden_size
            
        self.fc = Ensemble_layer(self.fc_input_size, self.output_size, self.dropout)

    def forward(self, fa_out, ta_out):
        
        # concat TA, FA representation
        fc_input = torch.cat([fa_out, ta_out], dim = -1) 
         
        # prediction layer
        fc_result = self.fc(fc_input.view(-1, self.fc_input_size))
        
        # output
        fc_result = fc_result.view(-1, self.sequence_length)
        
        return fc_result