import argparse

def get_Config():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='train', help='Action') # train / test
    
    # basic param
    parser.add_argument('--model_path', type=str, default='weights/', help='Model path')
    parser.add_argument('--path', type=str, default='./data/ship_paper_50_2.csv', help='Data path')  
    parser.add_argument('--date1', type=str, default='2019-05', help='date1') # validation date criteria
    parser.add_argument('--date2', type=str, default='2019-08', help='date2') # test date criteria
    parser.add_argument('--new_sequence', type=bool, default=False, help='Different length seq') # if try different sequence length: True
    
    # model param
    parser.add_argument('--input_size', type=int, default=16, help='Input size') 
    parser.add_argument('--output_size', type=int, default=1, help='Output size')
    parser.add_argument('--hidden_size', type=int, default=4, help='Hidden size')
    parser.add_argument('--sequence_length', type=int, default=50, help='Sequence length') 
    parser.add_argument('--num_layers', type=int, default=2, help='Num layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--out_type', type=str, default='representation', help='Ens input') 
    parser.add_argument('--function_type', type=str, default='sig', help='TA function') # if try different function type: sig, linear, quad, cubic, exp

    # learning param
    parser.add_argument('--epoch', type=int, default=100, help='Epoch')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--att_type', type=str, default='TA', help='Att type') # BASE, SA, TA, FA, ENS
    
    # masking param, (not applied in this version)
    parser.add_argument('--time_param', type=int, default=800, help='Time masking') # time masking
    parser.add_argument('--feature_param', type=float, default=0.9, help='Feature masking') # feature masking
    
    # device 
    parser.add_argument('--device', type=str, default='cuda:0', help='Gpu')
    parser.add_argument('--logging', type=bool, default=False, help='Logging option')
    
    arguments = parser.parse_args()
    
    return arguments