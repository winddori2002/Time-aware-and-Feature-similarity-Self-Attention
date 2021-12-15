from utils import *
import pandas as pd
import random
import os
import datetime
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pack_padded_sequence,pad_sequence, pad_packed_sequence

def get_data(path, date1, date2,sequence_length, new_sequence=False):
    
    ship2 = pd.read_csv(path, engine = 'python', encoding = 'cp949')
    ship2['UTC'] = pd.to_datetime(ship2['UTC'])
    ship2 = ship2.sort_values(['VESSELCODE','UTC'])
    ship2.index = list(range(len(ship2)))
    ship2['UTC_DIFF'] = ship2['UTC_DIFF'].astype('int')
    
    if new_sequence == True:
        ship2 = ship2.drop(columns = ['SEQUENCE2','UTC_DIFF'])
        ship2 = sequential_sorting(ship2, sequence_length)
        ship2 = ship2.drop(columns = ['UTC_LAG','SEQUENCE'])
        
    for col in ship2.columns:
        if ship2[col].dtype == 'float64':
            ship2[col] = ship2[col].astype(np.float32)
        elif ship2[col].dtype == 'int64':
            ship2[col] = ship2[col].astype(np.int32)   
            
    xtrain = ship2[ship2['UTC']< date1]
    xtrain.drop(columns = ['UTC','VESSELCODE','FOC','SEQUENCE2','UTC_DIFF'],inplace = True)
    train_mean = xtrain.describe().loc['mean']
    train_std = xtrain.describe().loc['std']
    
    xtest = ship2[ship2['UTC']< date2]
    xtest.drop(columns = ['UTC','VESSELCODE','FOC','SEQUENCE2','UTC_DIFF'],inplace = True)
    test_mean = xtest.describe().loc['mean']
    test_std = xtest.describe().loc['std']
    
    utc_train = ship2[ship2['UTC']< date1]['UTC_DIFF']
    train_utc_mean = np.mean(utc_train)
    train_utc_std = np.std(utc_train)
    
    utc_test = ship2[ship2['UTC']< date2]['UTC_DIFF']
    test_utc_mean = np.mean(utc_test)
    test_utc_std = np.std(utc_test)    
    
    return ship2, train_mean, train_std, test_mean, test_std, train_utc_mean, train_utc_std, test_utc_mean, test_utc_std
            
class ship_data(Dataset):
    def __init__(self, data, mean, std, u_mean,u_std,date1, date2):
        self.data = data
        self.mean = mean
        self.std = std
        self.u_mean = u_mean
        self.u_std = u_std        
        self.date1 = date1
        self.date2 = date2
        self.data['UTC'] = pd.to_datetime(self.data['UTC'])
    
        self.data = self.data[(self.data['UTC'] > self.date1) & (self.data['UTC'] < self.date2)]
        self.data = self.data.sort_values(['VESSELCODE','UTC'])
        self.data.index = list(range(len(self.data)))
        
        self.seq_idx = self.data[self.data['SEQUENCE2']==0].index
        self.time_diff = self.data['UTC_DIFF']
        self.X = self.data.drop(columns=['VESSELCODE', 'UTC', 'SEQUENCE2','FOC','UTC_DIFF'])
        self.time_diff = torch.from_numpy(self.time_diff.values)
        self.X = torch.from_numpy(((self.X - self.mean)/self.std).values)
        self.y = torch.from_numpy(self.data['FOC'].values)
        self.seq = make_sequence(self.X, self.y, self.time_diff, self.seq_idx)
        
    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, index):
        return self.seq[index]
    
def make_sequence(x,y,z, idx):
    inout_seq = []
    seq_index = idx
    
    for i in range(len(idx)):
        if i == len(idx)-1:
            x_seq = x[seq_index[i]:]
            y_seq = y[seq_index[i]:]
            z_seq = z[seq_index[i]:]
            inout_seq.append((x_seq,y_seq,z_seq ))
            break
    
        x_seq = x[seq_index[i]:seq_index[i+1]]
        y_seq = y[seq_index[i]:seq_index[i+1]]
        z_seq = z[seq_index[i]:seq_index[i+1]]
        inout_seq.append((x_seq, y_seq,z_seq))
    
    return inout_seq

def pad_collate(batch):

    xx, yy,zz = zip(*batch)
    x_lens = [len(x) for x in xx]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    zz_pad = pad_sequence(zz, batch_first=True, padding_value=0)
    
    return xx_pad, yy_pad, x_lens, zz_pad



