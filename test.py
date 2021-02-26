import pandas as pd
import numpy as np

import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_sequence, pad_packed_sequence
import argparse
from utils import *
from config import *
from models import *
from attention_modules import *

class Tester:
    def __init__(self, args):
        
        self.args = args
        self.model_setting = select_setting(args)
        self.model = self.select_model()
        
    def select_model(self):

        if self.args.att_type == 'ENS':
            model_setting = {'input_size': self.args.input_size,
                             'output_size': self.args.output_size,
                             'hidden_size': self.args.hidden_size,
                             'sequence_length': self.args.sequence_length,
                             'num_layers': self.args.num_layers,
                             'dropout': self.args.dropout}  

            model_setting_ens = {'input_size': self.args.input_size,
                             'output_size': self.args.output_size,
                             'hidden_size': self.args.hidden_size,
                             'sequence_length': self.args.sequence_length,
                             'num_layers': self.args.num_layers,
                             'dropout': self.args.dropout,    
                             'out_type': self.args.out_type}  

            ta_model = TA_MODEL(**model_setting).double().to(self.args.device)
            fa_model = FA_MODEL(**model_setting).double().to(self.args.device)
            model = Ensemble(**model_setting_ens).double().to(self.args.device)
            return ta_model, fa_model, model

        else:
            model_setting = {'input_size': self.args.input_size,
                             'output_size': self.args.output_size,
                             'hidden_size': self.args.hidden_size,
                             'sequence_length': self.args.sequence_length,
                             'num_layers': self.args.num_layers,
                             'dropout': self.args.dropout}  

            if self.args.att_type == 'BASE':

                model = BASE_MODEL(**model_setting).double().to(self.args.device)

            elif self.args.att_type == 'SA':

                model = SA_MODEL(**model_setting).double().to(self.args.device)

            elif self.args.att_type == 'TA':

                model = TA_MODEL(**model_setting).double().to(self.args.device)

            elif self.args.att_type == 'FA':

                model = FA_MODEL(**model_setting).double().to(self.args.device)

            return model     
        
    def test(self, data_loader):
    
        model = self.model
        checkpoint = torch.load(self.args.model_path + self.args.att_type +'.pth') 
        model.load_state_dict(checkpoint['state_dict'])

        print(self.args.att_type,''+'model testing')

        criterion = nn.MSELoss()

        model.eval()
        with torch.no_grad():

            val_loss = 0

            for j, v_data in enumerate(data_loader):

                x_val_data, y_val_data, v_time_data = (v_data[0].double().to(self.args.device), v_data[2]) , v_data[1].double(), v_data[3].double()
                x_val_data = x_val_data
                y_val_data = y_val_data.to(self.args.device)
                v_time_data = v_time_data.to(self.args.device)

                val_outputs = model(x_val_data,v_time_data)[0]
                v_masking = (y_val_data != 0)

                v_loss = criterion(val_outputs[v_masking], y_val_data[v_masking])
                val_loss += np.sqrt(v_loss.item())

        print("test loss: {:.4f}".format(val_loss /(j+1)))  

    #     neptune.log_metric('test loss', val_loss/(j+1))     
    #     neptune.stop()    


    def test_ens(self, data_loader):
        
        ta_model, fa_model, model = self.model
        checkpoint_ta = torch.load(self.args.model_path + 'TA.pth') 
        checkpoint_fa = torch.load(self.args.model_path + 'FA.pth') 
        checkpoint = torch.load(self.args.model_path + self.args.att_type +'.pth') 

        ta_model.load_state_dict(checkpoint_ta['state_dict'])
        fa_model.load_state_dict(checkpoint_fa['state_dict'])
        model.load_state_dict(checkpoint['state_dict'])

        criterion = nn.MSELoss()

        print(self.args.att_type,''+'model testing')

        model.eval()
        with torch.no_grad():

            val_loss = 0

            for j, v_data in enumerate(data_loader):

                x_val_data, y_val_data, v_time_data = (v_data[0].double().to(self.args.device), v_data[2]) , v_data[1].double(), v_data[3].double()
                x_val_data = x_val_data
                y_val_data = y_val_data.to(self.args.device)
                v_time_data = v_time_data.to(self.args.device)

                fa_out, ta_out = model_connection(fa_model, ta_model, x_val_data, v_time_data, self.args.out_type)
                val_outputs = model(fa_out, ta_out)

                v_masking = (y_val_data != 0)

                v_loss = criterion(val_outputs[v_masking], y_val_data[v_masking])
                val_loss += np.sqrt(v_loss.item())

        print("test loss: {:.4f}".format(val_loss /(j+1)))  

    #     neptune.log_metric('test loss', val_loss/(j+1))     
    #     neptune.stop()           