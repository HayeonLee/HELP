####################################################################################################
# HELP: hardware-adaptive efficient latency prediction for nas via meta-learning, NeurIPS 2021
# Hayeon Lee, Sewoong Lee, Song Chong, Sung Ju Hwang 
# github: https://github.com/HayeonLee/HELP, email: hayeon926@kaist.ac.kr
####################################################################################################

import os
import torch
from parser import get_parser
from help import HELP

def main(args):
    set_seed(args)
    args = set_gpu(args)
    args = set_path(args)
    
    print(f'==> mode is [{args.mode}] ...')
    model = HELP(args)

    if args.mode == 'meta-train':
        model.meta_train()
    elif args.mode == 'meta-test':
        model.test_predictor()
        
    elif args.mode == 'nas':
        model.nas()     

        
def set_seed(args):
    # Set the random seed for reproducible experiments
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def set_gpu(args):
    os.environ['CUDA_VISIBLE_DEVICES']= '-1' if args.gpu == None else args.gpu
    args.gpu = int(args.gpu)
    return args 

def set_path(args):
    args.data_path = os.path.join(
        args.main_path, 'data', args.search_space)
    args.save_path = os.path.join(
            args.save_path, args.search_space)        
    args.save_path = os.path.join(args.save_path, args.exp_name)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        if args.mode != 'nas':
            os.makedirs(os.path.join(args.save_path, 'checkpoint'))
    print(f'==> save path is [{args.save_path}] ...')   
    return args 

if __name__ == '__main__':
    main(get_parser())
