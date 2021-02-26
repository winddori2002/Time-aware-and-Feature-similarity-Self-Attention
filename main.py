import neptune
from dataset import *
from utils import *
from models import *
from attention_modules import *
from train import *
from test import *

import warnings
warnings.filterwarnings(action='ignore')

import argparse
from config import get_Config
args = get_Config()
print(vars(args))


# load data 
total_data, train_mean, train_std, test_mean, test_std, train_utc_mean, train_utc_std, test_utc_mean, test_utc_std  = get_data(args.path, args.date1, args.date2, 
                                                       args.sequence_length,args.new_sequence)

train = ship_data(total_data, train_mean, train_std, train_utc_mean, train_utc_std, '2013', args.date1)
val = ship_data(total_data, train_mean, train_std,  train_utc_mean, train_utc_std, args.date1, args.date2)
test = ship_data(total_data, test_mean, test_std,  test_utc_mean, test_utc_std, args.date2, '2020')
train_loader = DataLoader(train, batch_size = args.batch_size,collate_fn = pad_collate,shuffle=False)
val_loader = DataLoader(val, batch_size = args.batch_size,collate_fn = pad_collate, shuffle = False)
test_loader = DataLoader(test, batch_size = args.batch_size,collate_fn = pad_collate, shuffle = False)


# log setting / trainer, tester class 
trainer = Trainer(args)
tester = Tester(args)

PARAMS = {'epoch':args.epoch,
          'batch_size':args.batch_size,
          'lr':args.lr,
          'att_type':args.att_type,
          'device': args.device}
PARAMS.update(trainer.model_setting)

# experiment_name = 'Experiment'
# tag_name = ['Tag']
# neptune_load(experiment_name, tag_name, PARAMS)

if args.action == 'train':
    if args.att_type == 'ENS':
        trainer.train_ens(train_loader, val_loader)
        tester.test_ens(test_loader)
    else:
        trainer.train(train_loader, val_loader)
        tester.test(test_loader)
        
else:
    if args.att_type == 'ENS':
        tester.test_ens(test_loader)
    else:
        tester.test(test_loader)