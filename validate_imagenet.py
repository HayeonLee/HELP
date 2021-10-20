####################################################################################################
# HELP: hardware-adaptive efficient latency prediction for nas via meta-learning, NeurIPS 2021
# Hayeon Lee, Sewoong Lee, Song Chong, Sung Ju Hwang 
# github: https://github.com/HayeonLee/HELP, email: hayeon926@kaist.ac.kr
####################################################################################################
import os
import argparse
import math
import json 
import torch
from torchvision import transforms
from ofa.model_zoo import ofa_net
from ofa.imagenet_classification.run_manager import ImagenetRunConfig, RunManager

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='set visible gpus')
parser.add_argument('--config_path', type=str, default=None)
parser.add_argument('--imagenet_save_path', type=str, default=None)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

def build_val_transform(size):
        return transforms.Compose([
            transforms.Resize(int(math.ceil(size / 0.875))),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


config = json.load(open(args.config_path, 'r'))

ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)
run_config = ImagenetRunConfig(test_batch_size=100, n_worker=20, save_path=args.imagenet_save_path)
ofa_network.set_active_subnet(ks=config['ks'], e=config['e'], d=config['d'])
subnet = ofa_network.get_active_subnet(preserve_weight=True) 
run_config.data_provider.assign_active_img_size(config['r'])
run_manager = RunManager('.tmp/eval_subnet', subnet, run_config, init=False)

run_manager.reset_running_statistics(net=subnet)

loss, (top1, top5) = run_manager.validate(net=subnet)
print('Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (loss, top1, top5))
    
