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

class TimeAwareAttention(nn.Module):

    def __init__(self, d_model, dropout):
        super().__init__()

        self.temperature = (d_model**(0.5))
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_qt = nn.Linear(d_model, d_model, bias=False)
        self.w_kt = nn.Linear(d_model, d_model, bias=False)

        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.attention = TimeScaledDotProduct(self.temperature, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, inputs_lengths,distance_matrix):

        sz, max_len = q.size(0), q.size(1)
        q_t = q.clone()
        k_t = k.clone()
        residual = q.clone()

        # shape: b x len x d
        # linear projection
        q = self.w_q(q).view(sz, max_len, -1)
        k = self.w_k(k).view(sz, max_len, -1)
        v = self.w_v(v).view(sz, max_len, -1)
        q_t = self.w_qt(q_t).view(sz, max_len, -1)
        k_t = self.w_kt(k_t).view(sz, max_len, -1)
        
        # alpha t for transforming Qt Kt
        distance_weights = distance_matrix
        q, attn = self.attention(q, k, v, q_t, k_t, inputs_lengths,distance_weights)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn
    
class TimeScaledDotProduct(nn.Module):

    def __init__(self, temperature, attn_dropout):
        super().__init__()
        
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, q_t, k_t, inputs_lengths, distance_weights):

        # scaled dot product for self attention
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        
        # transform Qt, Kt
        q_t = torch.matmul(distance_weights, q_t)
        k_t = torch.matmul(distance_weights, k_t)
        # scaled dot
        distance_attn = torch.matmul(q_t / self.temperature, k_t.transpose(1, 2))
        
        # combine with self attention
        attn = attn + distance_attn
        attn = pad_zero_masking(attn, inputs_lengths)
        attn = torch.tanh(attn)
        attn = pad_zero_masking(attn, inputs_lengths)
        output = torch.matmul(attn, v)

        return output, attn
    
class FeatureSimilarityAttention(nn.Module):

    def __init__(self, d_model, dropout):
        super().__init__()

        self.temperature = (d_model**(0.5))
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_qf = nn.Linear(d_model, d_model, bias=False)
        self.w_kf = nn.Linear(d_model, d_model, bias=False)

        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.FS = FeatureSim(self.temperature,dropout)
        self.attention = FeatureScaledDotProduct(self.temperature, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, inputs_lengths,x):

        sz, max_len = q.size(0), q.size(1)
        q_f = q.clone()
        k_f = k.clone()
        residual = q.clone()

        # shape: b x len x d
        # linear projection
        q = self.w_q(q).view(sz, max_len, -1)
        k = self.w_k(k).view(sz, max_len, -1)
        v = self.w_v(v).view(sz, max_len, -1)
        q_f = self.w_qf(q_f).view(sz, max_len, -1)
        k_f = self.w_kf(k_f).view(sz, max_len, -1)
        
        # alpha f for transforming Qf, Kf
        # feature similarity weight
        feature_attn = self.FS(x, inputs_lengths)
        q, attn = self.attention(q, k, v, q_f, k_f, inputs_lengths,feature_attn)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn
    
class FeatureScaledDotProduct(nn.Module):

    def __init__(self, temperature, attn_dropout):
        super().__init__()
        
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, q_f, k_f, inputs_lengths, feature_attn):

        # self attention
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        
        # feature attention
        q_f = torch.matmul(feature_attn, q_f)
        k_f = torch.matmul(feature_attn, k_f)
        feature_attn = torch.matmul(q_f / self.temperature, k_f.transpose(1, 2))
        
        # combine with self attention
        attn = attn + feature_attn
        attn = pad_zero_masking(attn, inputs_lengths)
        attn = torch.tanh(attn)
        attn = pad_zero_masking(attn, inputs_lengths)
        output = torch.matmul(attn, v)

        return output, attn

class FeatureSim(nn.Module):
    def __init__(self, temperature, dropout):
        super(FeatureSim, self).__init__()
        
        self.temperature = temperature
        self.feature_importance =  nn.Parameter(torch.zeros(11, device = device), requires_grad=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, x_lengths):

        # except spec features
        x_feature = x[:,:,:11]
        # abs L1 distance
        feature_distance = torch.abs(x_feature.unsqueeze(2)-x_feature.unsqueeze(1))

        # weighted sum with feature importance
        feature_distance = feature_distance * self.feature_importance
        feature_attn = torch.sum(feature_distance, dim = -1)
        feature_attn = pad_masking(feature_attn, x_lengths)
        feature_attn = (F.softmax(feature_attn, dim = -1))
        feature_attn = feature_attn.masked_fill(torch.isnan(feature_attn), 0)   

        return feature_attn
    
class SingleHeadAttention(nn.Module):

    def __init__(self, d_model, dropout):
        super().__init__()

        self.temperature = (d_model**(0.5))
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.attention = ScaledDotProduct(self.temperature, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, inputs_lengths):

        sz, max_len = q.size(0), q.size(1)
        residual = q.clone()

        # shape: b x len x d
        q = self.w_q(q).view(sz, max_len, -1)
        k = self.w_k(k).view(sz, max_len, -1)
        v = self.w_v(v).view(sz, max_len, -1)

        q, attn = self.attention(q, k, v, inputs_lengths)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn
    
class ScaledDotProduct(nn.Module):

    def __init__(self, temperature, attn_dropout):
        super().__init__()
        
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, inputs_lengths):

        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        attn_masked = pad_masking(attn, inputs_lengths)
        attn_masked = self.dropout(F.softmax(attn_masked, dim=-1))
        attn_masked = attn_masked.masked_fill(torch.isnan(attn_masked), 0)
        output = torch.matmul(attn_masked, v)

        return output, attn_masked
