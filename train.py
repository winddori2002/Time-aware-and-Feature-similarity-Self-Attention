import numpy as np
import datetime

import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
import argparse
from utils import *
from config import *
from models import *
from attention_modules import *

class Trainer:
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
        
    def train(self, train_loader, val_loader):
        
        model = self.model
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

        print(self.args.att_type,''+'model training')

        best_loss = 1000000

        for num_epochs in range(self.args.epoch):
            train_loss = 0

            model.train()
            for i,t_data in enumerate(train_loader):
                x_data, y_data, time_data = (t_data[0].double().to(self.args.device), t_data[2]), t_data[1].double() , t_data[3].double()
                x_data = x_data
                y_data = y_data.to(self.args.device)
                time_data = time_data.to(self.args.device)
                optimizer.zero_grad()
                outputs = model(x_data, time_data)[0]
                
                # masking for not including loss for pad values
                masking = (y_data != 0)

                loss = criterion(outputs[masking], y_data[masking])
                loss.backward()
                optimizer.step()

                train_loss += np.sqrt(loss.item())


            model.eval()
            with torch.no_grad():

                val_loss = 0

                for j, v_data in enumerate(val_loader):

                    x_val_data, y_val_data, v_time_data = (v_data[0].double().to(self.args.device), v_data[2]) , v_data[1].double(), v_data[3].double()
                    x_val_data = x_val_data
                    y_val_data = y_val_data.to(self.args.device)
                    v_time_data = v_time_data.to(self.args.device)

                    val_outputs = model(x_val_data,v_time_data)[0]
                    v_masking = (y_val_data != 0)

                    v_loss = criterion(val_outputs[v_masking], y_val_data[v_masking])
                    val_loss += np.sqrt(v_loss.item())


            print("epoch: {}/{}  | trn loss: {:.4f} | val loss: {:.4f}".format(self.args.epoch, num_epochs+1, train_loss /(i+1), val_loss /(j+1)))  

            if  val_loss /(j+1) < best_loss:
                best_loss =  val_loss /(j+1)
                checkpoint = {'loss': val_loss /(j+1),
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, self.args.model_path + self.args.att_type + '.pth')    

    #         neptune.log_metric('train loss', train_loss /(i+1))
    #         neptune.log_metric('val loss', val_loss/(j+1))     


    def train_ens(self, train_loader, val_loader):
        
        ta_model, fa_model, model = self.model
        checkpoint_ta = torch.load(self.args.model_path + 'TA.pth') 
        checkpoint_fa = torch.load(self.args.model_path + 'FA.pth') 

        ta_model.load_state_dict(checkpoint_ta['state_dict'])
        fa_model.load_state_dict(checkpoint_fa['state_dict'])  

        print(self.args.att_type,''+'model training')

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

        best_loss = 1000000

        for num_epochs in range(self.args.epoch):
            train_loss = 0

            model.train()
            for i,t_data in enumerate(train_loader):
                x_data, y_data, time_data = (t_data[0].double().to(self.args.device), t_data[2]), t_data[1].double() , t_data[3].double()
                x_data = x_data
                y_data = y_data.to(self.args.device)
                time_data = time_data.to(self.args.device)
                optimizer.zero_grad()
                
                # model connection module to link TA-FA with ENS
                fa_out, ta_out = model_connection(fa_model, ta_model, x_data, time_data, self.args.out_type)
                outputs = model(fa_out, ta_out)
                masking = (y_data != 0)

                loss = criterion(outputs[masking], y_data[masking])
                loss.backward()
                optimizer.step()

                train_loss += np.sqrt(loss.item())


            model.eval()
            with torch.no_grad():

                val_loss = 0

                for j, v_data in enumerate(val_loader):

                    x_val_data, y_val_data, v_time_data = (v_data[0].double().to(self.args.device), v_data[2]) , v_data[1].double(), v_data[3].double()
                    x_val_data = x_val_data
                    y_val_data = y_val_data.to(self.args.device)
                    v_time_data = v_time_data.to(self.args.device)

                    fa_out, ta_out = model_connection(fa_model, ta_model, x_val_data, v_time_data, self.args.out_type)
                    val_outputs = model(fa_out, ta_out)

                    v_masking = (y_val_data != 0)

                    v_loss = criterion(val_outputs[v_masking], y_val_data[v_masking])
                    val_loss += np.sqrt(v_loss.item())

            print("epoch: {}/{}  | trn loss: {:.4f} | val loss: {:.4f}".format(self.args.epoch, num_epochs+1, train_loss /(i+1), val_loss /(j+1)))  

            if  val_loss /(j+1) < best_loss:
                best_loss =  val_loss /(j+1)
                checkpoint = {'loss': val_loss /(j+1),
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, self.args.model_path + self.args.att_type + '.pth')    

    #         neptune.log_metric('train loss', train_loss /(i+1))
    #         neptune.log_metric('val loss', val_loss/(j+1))    



