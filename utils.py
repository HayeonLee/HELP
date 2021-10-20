####################################################################################################
# HELP: hardware-adaptive efficient latency prediction for nas via meta-learning, NeurIPS 2021
# Hayeon Lee, Sewoong Lee, Song Chong, Sung Ju Hwang 
# github: https://github.com/HayeonLee/HELP, email: hayeon926@kaist.ac.kr
####################################################################################################
import logging
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau, pearsonr
import copy
#import wandb


def get_minmax_latency_index(meta_train_devices, train_idx, latency):
    rank = {}
    cnt = {}
    for device in meta_train_devices:
        lat, rank_idx = torch.sort(latency[device][train_idx[device]])
        for r, t in zip(rank_idx, train_idx[device]):
            t = t.item()
            if not t in rank.keys():
                rank[t] = 0
                cnt[t] = 0
            rank[t] += r
            cnt[t] += 1

    max_lat_rank = -10000000
    max_lat_idx = None
    min_lat_rank = 100000000
    min_lat_idx = None
    for (t, r), c in zip(rank.items(), cnt.values()):
        if c < len(meta_train_devices):
            continue
        if r > max_lat_rank:
            max_lat_rank = r
            max_lat_idx = t 
        if r < min_lat_rank:
            min_lat_rank = r 
            min_lat_idx = t
    return max_lat_idx, min_lat_idx


def log_prob(dist, groundtruth):
    log_p = dist.log_prob(groundtruth)
    return -log_p.mean()

loss_fn = {
            'mse': lambda yq_hat, yq,: F.mse_loss(yq_hat, yq),
            'logprob': lambda yq_hat, yq: log_prob(dist, yq)
            }

def flat(v):
    if torch.is_tensor(v):
        return v.detach().cpu().numpy().reshape(-1)
    else:
        return v.reshape(-1)

metrics_fn = {
            'spearman': lambda yq_hat, yq: spearmanr(flat(yq_hat), flat(yq)),
            'pearsonr': lambda yq_hat, yq: pearsonr(flat(yq_hat), flat(yq)),
            'kendalltau': lambda yq_hat, yq: kendalltau(flat(yq_hat), flat(yq))
            }


class Log():
    def __init__(self, save_path, summary_steps, metrics, devices, split, writer=None, use_wandb=False):
        self.save_path = save_path
        self.metrics = metrics
        self.devices = devices
        self.summary_steps = summary_steps
        self.split = split
        self.writer = writer

        self.epi = []
        self.elems = {}
        for metric in metrics:  
            self.elems[metric] = { device: [] for device in devices }
        self.elems['loss'] = { device: [] for device in devices }
        self.elems['mse_loss'] = { device: [] for device in devices }
        self.elems['kl_loss'] = { device: [] for device in devices }
        # self.elems['denorm_mse'] = { device: [] for device in devices }

        self.use_wandb = use_wandb

    def update_epi(self, i_epi):
        self.epi.append(i_epi)

    def update(self, i_epi, metric, device, val):
        self.elems[metric][device].append(val)
        if self.use_wandb:
            log_dict = {f'{self.split}_{metric}/{device}': val}
            wandb.log(log_dict, step=i_epi)
        if self.writer is not None:
            self.writer.add_scalar(f'{self.split}_{metric}/{device}', val, i_epi)  

    def avg(self, i_epi, metric, is_print=True):
        v = 0.0
        cnt = 0
        for device in self.devices:
            v += self.get(metric, device, i_epi)
            cnt += 1     
        if self.use_wandb:
            log_dict = {f'mean/{self.split}_{metric}': v / cnt}
            wandb.log(log_dict, step=i_epi)
        if self.writer is not None and is_print:
            self.writer.add_scalar(f'mean/{self.split}_{metric}', v / cnt, i_epi)
        return v / cnt
    
    # def last(self, metric, device):
    #     return self.elems[metric][device][-1]

    def get(self, metric, device, i_epi):
        idx = self.epi.index(i_epi)
        return self.elems[metric][device][idx]

    def save(self):
        torch.save({
                    'summary_steps': self.summary_steps,
                    'episode': self.epi,
                    'elems': self.elems
                    }, 
                    os.path.join(self.save_path, f'{self.split}_log_data.pt'))


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, 'w')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

def denorm(lat, maxv, minv):
    return lat * (maxv-minv) + minv

def normalization(latency, index=None, portion=0.9):
    if index != None:
        min_val = min(latency[index])
        max_val = max(latency[index])
    else :
        min_val = min(latency)
        max_val = max(latency)
    latency = (latency - min_val) / (max_val - min_val) * portion + (1 - portion) / 2
    return latency

def arch_enc(arch):
    feature=[]
    for i in arch:
        onehot = np.zeros(6)
        if i == 8 :
            feature = np.hstack([feature, onehot])
        else :
            if i < 4:
                onehot[0] = 1
            elif i < 8:
                onehot[1] = 1
            k = i % 4
            onehot[2+k] = 1
            feature = np.hstack([feature, onehot])
    assert len(feature) == 132
    return torch.FloatTensor(feature)
    
    
def add_global_node( mx, ifAdj):
    """add a global node to operation or adjacency matrixs, fill diagonal for adj and transpose adjs"""
    if (ifAdj):
        mx = np.column_stack((mx, np.ones(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        np.fill_diagonal(mx, 1)
        mx = mx.T
    else:
        mx = np.column_stack((mx, np.zeros(mx.shape[0], dtype=np.float32)))
        mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
        mx[mx.shape[0] - 1][mx.shape[1] - 1] = 1
    return torch.FloatTensor(mx)

def padzero( mx, ifAdj, maxsize=7):
    if ifAdj:
        while mx.shape[0] < maxsize:
            mx = np.column_stack((mx, np.zeros(mx.shape[0], dtype=np.float32)))
            mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
    else:
        while mx.shape[0] < maxsize:
            mx = np.row_stack((mx, np.zeros(mx.shape[1], dtype=np.float32)))
    return mx


def arch_encoding_ofa(arch):
    # This function converts a network config to a feature vector (128-D).
    ks_list, ex_list, d_list, r = copy.deepcopy(arch['ks']), copy.deepcopy(arch['e']), copy.deepcopy(arch['d']), arch['r']
    
    ks_map = {}
    ks_map[3]=0
    ks_map[5]=1
    ks_map[7]=2
    ex_map = {}
    ex_map[3]=0
    ex_map[4]=1
    ex_map[6]=2
    
    
    start = 0
    end = 4
    for d in d_list:
        for j in range(start+d, end):
            ks_list[j] = 0
            ex_list[j] = 0
        start += 4
        end += 4

    # convert to onehot
    ks_onehot = [0 for _ in range(60)]
    ex_onehot = [0 for _ in range(60)]
    r_onehot = [0 for _ in range(25)] #128 ~ 224

    for i in range(20):
        start = i * 3
        if ks_list[i] != 0:
            ks_onehot[start + ks_map[ks_list[i]]] = 1
        if ex_list[i] != 0:
            ex_onehot[start + ex_map[ex_list[i]]] = 1

    r_onehot[(r - 128) // 4] = 1
    return torch.Tensor(ks_onehot + ex_onehot + r_onehot)


def data_norm(v, src, des):
    min_s = min(src)
    max_s = max(src)
    min_d = min(des)
    max_d = max(des)
    nv = (v-min_s) / (max_s-min_s)
    nv = nv *(max_d-min_d) + min_d
    return nv
