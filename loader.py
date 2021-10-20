####################################################################################################
# HELP: hardware-adaptive efficient latency prediction for nas via meta-learning, NeurIPS 2021
# Hayeon Lee, Sewoong Lee, Song Chong, Sung Ju Hwang 
# github: https://github.com/HayeonLee/HELP, email: hayeon926@kaist.ac.kr
####################################################################################################
import os
import random
import numpy as np
import torch

from utils import *


class Data:
    def __init__(self, mode,
                    data_path, 
                    search_space,
                    meta_train_devices,
                    meta_valid_devices, 
                    meta_test_devices,
                    num_inner_tasks, 
                    num_meta_train_sample,
                    num_sample, 
                    num_query,
                    sampled_arch_path,
                    num_query_meta_train_task=200,
                    remove_outlier=True):
        self.mode = mode
        self.data_path = data_path
        self.search_space = search_space
        self.meta_train_devices = meta_train_devices
        self.meta_valid_devices = meta_valid_devices
        self.meta_test_devices = meta_test_devices
        self.num_inner_tasks = num_inner_tasks
        self.num_meta_train_sample = num_meta_train_sample
        self.num_sample = num_sample
        self.num_query = num_query
        self.num_query_meta_train_task = num_query_meta_train_task
        self.remove_outlier = remove_outlier
        # Latency-constrainted NAS
        self.sampled_arch_path = sampled_arch_path
        
        self.load_archs()

        self.train_idx ={}
        self.valid_idx = {}
        self.latency = {}
        self.norm_latency = {}
        nts = self.num_meta_train_sample
        for device in meta_train_devices + meta_valid_devices + meta_test_devices:
            self.latency[device] = torch.FloatTensor(
                torch.load(os.path.join(data_path, 'latency', f'{device}.pt')))
            train_idx = torch.arange(len(self.archs))[:nts]
            valid_idx = torch.arange(len(self.archs))[nts:nts+self.num_query]

            if self.remove_outlier:
                self.train_idx[device] = train_idx[
                    np.argsort(self.latency[device][train_idx])[
                        int(len(train_idx)*0.1):int(len(train_idx)*0.9)]]
                self.valid_idx[device] = valid_idx[
                    np.argsort(self.latency[device][valid_idx])[
                        int(len(valid_idx)*0.1):int(len(valid_idx)*0.9)]]

            self.norm_latency[device] = normalization(
                                        latency=self.latency[device],
                                        index = self.train_idx[device]
                                        )
        # load index set of reference architectures
        self.hw_emb_idx = torch.load(
            os.path.join(data_path, 'hardware_embedding_index.pt'))

        if self.mode == 'nas':
            self.max_lat_idx, self.min_lat_idx = get_minmax_latency_index(
                meta_train_devices + meta_valid_devices, self.train_idx, self.latency)
            self.nas_norm_latency = {}
            for device in meta_train_devices + meta_valid_devices + meta_test_devices:
                self.nas_norm_latency[device] = normalization(
                    latency=self.latency[device],
                    index = torch.tensor([self.max_lat_idx, self.min_lat_idx] + self.hw_emb_idx))
            if self.search_space == 'nasbench201':
                self._load_arch_candidates()
                self._load_arch_str2idx()

        print('==> load data ...')


    def load_archs(self):
        if self.search_space == 'nasbench201':
            # operations and adjacency matrix of neural architectures in NAS-Bench-201
            self.archs = [[add_global_node(_['operation'], ifAdj=False), 
                            add_global_node(_['adjacency_matrix'],ifAdj=True)]
                for _ in torch.load(os.path.join(self.data_path, 'architecture.pt'))]
        elif self.search_space == 'fbnet':
            self.archs = [arch_enc(_['op_idx_list']) for _ in 
                torch.load(os.path.join(self.data_path, 'metainfo.pt'))['arch']]
        elif self.search_space == 'ofa':
            self.archs = [arch_encoding_ofa(arch) for arch in 
                torch.load(os.path.join(self.data_path, 'ofa_archs.pt'))['arch']]


    def generate_episode(self):
        # metabatch
        episode = []

        # meta-batch
        rand_device_idx = torch.randperm(
                            len(self.meta_train_devices))[:self.num_inner_tasks]
        for t in rand_device_idx:
            # sample devices
            device = self.meta_train_devices[t]
            # hardware embedding
            latency = self.latency[device]
            hw_embed = latency[self.hw_emb_idx]
            hw_embed = normalization(hw_embed, portion=1.0)

            # samples for finetuning & test (query)
            rand_idx = self.train_idx[device][torch.randperm(len(self.train_idx[device]))]
            finetune_idx = rand_idx[:self.num_sample]
            qry_idx = rand_idx[self.num_sample:self.num_sample+self.num_query_meta_train_task]

            if self.search_space in ['fbnet', 'ofa']:
                x_finetune = torch.stack([self.archs[_] for _ in finetune_idx])
                x_qry = torch.stack([self.archs[_] for _ in qry_idx])

            elif self.search_space == 'nasbench201':
                x_finetune = [torch.stack([self.archs[_][0] for _ in finetune_idx]),
                            torch.stack([self.archs[_][1] for _ in finetune_idx])]
                x_qry = [torch.stack([self.archs[_][0] for _ in qry_idx]), 
                        torch.stack([self.archs[_][1] for _ in qry_idx])]
            y_finetune = self.norm_latency[device][finetune_idx].view(-1, 1)
            y_qry = self.norm_latency[device][qry_idx].view(-1, 1)

            episode.append((hw_embed, x_finetune, y_finetune, x_qry, y_qry, device))

        return episode

    def generate_test_tasks(self, split=None):
        if split == 'meta_train':
            device_list = self.meta_train_devices
        elif split == 'meta_valid':
            device_list = self.meta_valid_devices
        elif split == 'meta_test':
            device_list = self.meta_test_devices
        else: NotImplementedError

        tasks = []
        for device in device_list:
            tasks.append(self.get_task(device))
        return tasks

    def get_task(self, device=None, num_sample=None):
        if num_sample == None:
            num_sample = self.num_sample
    
        latency = self.latency[device]
        # hardware embedding
        hw_embed = latency[self.hw_emb_idx]
        hw_embed = normalization(hw_embed, portion=1.0)        
        
        # samples for finetuing & test (query)
        rand_idx = self.train_idx[device][torch.randperm(len(self.train_idx[device]))]
        finetune_idx = rand_idx[:num_sample]

        if self.search_space in ['fbnet', 'ofa']:
            x_finetune = torch.stack([self.archs[_] for _ in finetune_idx])
            x_qry = torch.stack([self.archs[_] for _ in self.valid_idx[device]])
        elif self.search_space == 'nasbench201':
            x_finetune = [torch.stack([self.archs[_][0] for _ in finetune_idx]),
                        torch.stack([self.archs[_][1] for _ in finetune_idx])]
            x_qry = [torch.stack([self.archs[_][0] for _ in self.valid_idx[device]]),
                    torch.stack([self.archs[_][1] for _ in self.valid_idx[device]])]
        
        y_finetune = self.norm_latency[device][finetune_idx].view(-1, 1)
        y_qry = self.norm_latency[device][self.valid_idx[device]].view(-1, 1)

        return hw_embed, x_finetune, y_finetune, x_qry, y_qry, device

    def _load_arch_candidates(self):
        # architecture candidates obtained by MetaD2A
        loaded = open(self.sampled_arch_path, 'r')
        self.arch_candidates = {
                                'arch': [],
                                'true_acc': []
                               }
        for line in loaded.readlines()[1:]:
            arch, true_acc, _ = line.split(',')
            self.arch_candidates['arch'].append(arch)
            self.arch_candidates['true_acc'].append(true_acc)


    def _load_arch_str2idx(self):
        self.arch_str2idx = torch.load(os.path.join(self.data_path, 'str_arch2idx.pt'))


    def get_nas_task(self, device=None):
        num_sample = self.num_sample
        latency = self.latency[device]
        # hardware embedding
        hw_embed = latency[self.hw_emb_idx]
        hw_embed = normalization(hw_embed, portion=1.0)        

        if self.search_space in 'ofa':
            # samples for finetuning & test (query)
            finetune_idx = self.hw_emb_idx
            norm_latency = self.nas_norm_latency[device]
            x_finetune = torch.stack([self.archs[_] for _ in finetune_idx])
        elif self.search_space == 'nasbench201':
            # samples for finetuning & test (query)
            rand_idx = torch.randperm(len(self.train_idx[device]))
            finetune_idx = self.train_idx[device][rand_idx[:num_sample]]
            norm_latency = self.nas_norm_latency[device]
            x_finetune = [torch.stack([self.archs[_][0] for _ in finetune_idx]),
                        torch.stack([self.archs[_][1] for _ in finetune_idx])]
        y_finetune = norm_latency[finetune_idx].view(-1, 1)
        y_finetune_gt = latency[finetune_idx].view(-1, 1)

        if self.search_space == 'nasbench201':
            # architecture candidates obtained by MetaD2A
            metad2a_idx = [self.arch_str2idx[_] for _ in self.arch_candidates['arch']]
            x_qry = [torch.stack([self.archs[_][0] for _ in metad2a_idx]),
                    torch.stack([self.archs[_][1] for _ in metad2a_idx])]
            y_qry = norm_latency[metad2a_idx].view(-1, 1)   
            y_qry_gt = latency[metad2a_idx].view(-1, 1)     
            return hw_embed, x_finetune, y_finetune, x_qry, y_qry, device, y_finetune_gt, y_qry_gt
        elif self.search_space == 'ofa':
            return hw_embed, x_finetune, y_finetune, y_finetune_gt

    # def get_nas_task(self, device=None):
    #     num_sample = self.num_sample
    #     latency = self.latency[device]
    #     # hardware embedding
    #     hw_embed = latency[self.hw_emb_idx]
    #     hw_embed = normalization(hw_embed, portion=1.0)        
        
    #     # samples for finetuning & test (query)
    #     rand_idx = torch.randperm(len(self.train_idx[device]))
    #     finetune_idx = self.train_idx[device][rand_idx[:num_sample]]
    #     norm_latency = self.nas_norm_latency[device]

    #     x_finetune = [torch.stack([self.archs[_][0] for _ in finetune_idx]),
    #                  torch.stack([self.archs[_][1] for _ in finetune_idx])]
    #     y_finetune = norm_latency[finetune_idx].view(-1, 1)
    #     y_finetune_gt = latency[finetune_idx].view(-1, 1)

    #     if self.search_space == 'nasbench201':
    #         # architecture candidates obtained by MetaD2A
    #         metad2a_idx = [self.arch_str2idx[_] for _ in self.arch_candidates['arch']]
    #         x_qry = [torch.stack([self.archs[_][0] for _ in metad2a_idx]),
    #                 torch.stack([self.archs[_][1] for _ in metad2a_idx])]
    #         y_qry = norm_latency[metad2a_idx].view(-1, 1)   
    #         y_qry_gt = latency[metad2a_idx].view(-1, 1)     
    #     else:
    #         x_qry, y_qry, y_qry_gt = None, None, None
    #     return hw_embed, x_finetune, y_finetune, x_qry, y_qry, device, y_finetune_gt, y_qry_gt
