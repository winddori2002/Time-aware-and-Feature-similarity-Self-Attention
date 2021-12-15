import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
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
        criterion2 = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        print(self.args.att_type,''+'model training')

        best_loss = 1000000
        for num_epochs in range(self.args.epoch):
            train_loss = 0
            train_mae = 0

            model.train()
            for i, data in enumerate(train_loader):
                x_data, y_data, time_data = (data[0].double().to(self.args.device), data[2]), data[1].double(), data[3].double()
                x_data = x_data
                y_data = y_data.to(self.args.device)
                time_data = time_data.to(self.args.device)
                optimizer.zero_grad()
                outputs = model(x_data, time_data)[0]
                
                # masking for not including loss for pad values
                masking = (y_data != 0)

                loss = criterion(outputs[masking], y_data[masking])
                loss2 = criterion2(outputs[masking], y_data[masking])
                loss.backward()
                optimizer.step()

                train_loss += np.sqrt(loss.item())
                train_mae += loss2.item()

            model.eval()
            with torch.no_grad():

                val_loss = 0
                val_mae = 0

                for j, data in enumerate(val_loader):

                    x_data, y_data, time_data = (data[0].double().to(self.args.device), data[2]) , data[1].double(), data[3].double()
                    x_data = x_data
                    y_data = y_data.to(self.args.device)
                    time_data = time_data.to(self.args.device)

                    outputs = model(x_data,time_data)[0]
                    masking = (y_data != 0)

                    v_loss = criterion(outputs[masking], y_data[masking])
                    v_loss2 = criterion2(outputs[masking], y_data[masking])
                    val_loss += np.sqrt(v_loss.item())
                    val_mae += v_loss2.item()
                    
            print("epoch: {}/{}  | trn loss: {:.4f} | val loss: {:.4f}".format(self.args.epoch, num_epochs+1, train_loss /(i+1), val_loss /(j+1)))
            print("epoch: {}/{}  | trn mae: {:.4f} | val mae: {:.4f}".format(self.args.epoch, num_epochs+1, train_mae /(i+1), val_mae /(j+1)))  
            
            if  val_loss /(j+1) < best_loss:
                best_loss =  val_loss /(j+1)
                checkpoint = {'loss': val_loss /(j+1),
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, self.args.model_path + self.args.att_type + '.pth')    
                
            if self.args.logging:
                neptune.log_metric('train loss', train_loss /(i+1))
                neptune.log_metric('val loss', val_loss/(j+1))  
                neptune.log_metric('train mae', train_mae /(i+1))
                neptune.log_metric('val mae', val_mae/(j+1))  

    def train_ens(self, train_loader, val_loader):
        
        ta_model, fa_model, model = self.model
        checkpoint_ta = torch.load(self.args.model_path + 'TA.pth') 
        checkpoint_fa = torch.load(self.args.model_path + 'FA.pth') 

        ta_model.load_state_dict(checkpoint_ta['state_dict'])
        fa_model.load_state_dict(checkpoint_fa['state_dict'])  
        print(self.args.att_type,''+'model training')

        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        
        best_loss = 1000000
        for num_epochs in range(self.args.epoch):
            train_loss = 0
            train_mae = 0
            
            model.train()
            for i, data in enumerate(train_loader):
                x_data, y_data, time_data = (data[0].double().to(self.args.device), data[2]), data[1].double() , data[3].double()
                x_data = x_data
                y_data = y_data.to(self.args.device)
                time_data = time_data.to(self.args.device)
                optimizer.zero_grad()
                
                # model connection module to link TA-FA with ENS
                fa_out, ta_out = model_connection(fa_model, ta_model, x_data, time_data, self.args.out_type)
                outputs = model(fa_out, ta_out)
                masking = (y_data != 0)

                loss = criterion(outputs[masking], y_data[masking])
                loss2 = criterion2(outputs[masking], y_data[masking])
                loss.backward()
                optimizer.step()

                train_loss += np.sqrt(loss.item())
                train_mae += loss2.item()


            model.eval()
            with torch.no_grad():

                val_loss = 0
                val_mae = 0

                for j, data in enumerate(val_loader):

                    x_data, y_data, time_data = (data[0].double().to(self.args.device), data[2]) , data[1].double(), data[3].double()
                    x_data = x_data
                    y_data = y_data.to(self.args.device)
                    time_data = time_data.to(self.args.device)

                    fa_out, ta_out = model_connection(fa_model, ta_model, x_data, time_data, self.args.out_type)
                    outputs = model(fa_out, ta_out)

                    masking = (y_data != 0)

                    v_loss = criterion(outputs[masking], y_data[masking])
                    v_loss2 = criterion2(outputs[masking], y_data[masking])
                    val_loss += np.sqrt(v_loss.item())
                    val_mae += v_loss2.item()

            print("epoch: {}/{}  | trn loss: {:.4f} | val loss: {:.4f}".format(self.args.epoch, num_epochs+1, train_loss /(i+1), val_loss /(j+1)))  
            print("epoch: {}/{}  | trn mae: {:.4f} | val mae: {:.4f}".format(self.args.epoch, num_epochs+1, train_mae /(i+1), val_mae /(j+1)))  

            if  val_loss /(j+1) < best_loss:
                best_loss =  val_loss /(j+1)
                checkpoint = {'loss': val_loss /(j+1),
                              'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, self.args.model_path + self.args.att_type + '.pth')    

            if self.args.logging:
                neptune.log_metric('train loss', train_loss /(i+1))
                neptune.log_metric('val loss', val_loss/(j+1))  
                neptune.log_metric('train mae', train_mae /(i+1))
                neptune.log_metric('val mae', val_mae/(j+1))  




