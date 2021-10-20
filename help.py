####################################################################################################
# HELP: hardware-adaptive efficient latency prediction for nas via meta-learning, NeurIPS 2021
# Hayeon Lee, Sewoong Lee, Song Chong, Sung Ju Hwang 
# github: https://github.com/HayeonLee/HELP, email: hayeon926@kaist.ac.kr
####################################################################################################
import os
import logging
from collections import OrderedDict
from collections import defaultdict
import csv
from tqdm import tqdm
import json
#import wandb

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from net import MetaLearner
from net import Net
from net import InferenceNetwork
from loader import Data
from utils import *



class HELP:
    def __init__(self, args):
        self.args = args
        self.mode = args.mode
        self.metrics = args.metrics
        self.search_space = args.search_space
        self.load_path = args.load_path
        # Log
        self.save_summary_steps = args.save_summary_steps
        self.save_path = args.save_path
        # Data & Meta-learning Settings
        self.meta_train_devices = args.meta_train_devices
        self.meta_valid_devices = args.meta_valid_devices
        self.meta_test_devices = args.meta_test_devices
        self.num_inner_tasks = args.num_inner_tasks
        self.meta_lr = args.meta_lr
        self.num_episodes = args.num_episodes
        self.num_train_updates = args.num_train_updates
        self.num_eval_updates = args.num_eval_updates
        self.alpha_on = args.alpha_on
        self.inner_lr = args.inner_lr
        self.second_order = args.second_order
        # Meta-learner
        self.hw_emb_dim = args.hw_embed_dim
        self.layer_size = args.layer_size
        # Inference Network
        self.z_on = args.z_on
        self.determ = args.determ
        self.kl_scaling = args.kl_scaling
        self.z_scaling = args.z_scaling
        self.mc_sampling = args.mc_sampling
        # End to End NAS
        if self.mode == 'nas' and not self.search_space in ['nasbench201', 'ofa']: 
            raise NotImplementedError
        self.nas_target_device = args.nas_target_device
        self.latency_constraint = args.latency_constraint
        # Data
        self.data = Data(args.mode,
                        args.data_path, 
                        args.search_space,
                        args.meta_train_devices, 
                        args.meta_valid_devices,
                        args.meta_test_devices,
                        args.num_inner_tasks, 
                        args.num_meta_train_sample,
                        args.num_samples, 
                        args.num_query,
                        args.sampled_arch_path)
        # Model
        self.model = MetaLearner(args.search_space, 
                                 args.hw_embed_on,
                                 args.hw_embed_dim,
                                 args.layer_size).cuda()
        self.model_params = list(self.model.parameters())
        if self.alpha_on:
            self.define_task_lr_params()
            self.model_params += list(self.task_lr.values())
        else: self.task_lr = None
        
        if self.z_on:
            self.inference_network = InferenceNetwork(args.hw_embed_on,
                                        args.hw_embed_dim,
                                        args.layer_size,
                                        args.determ).cuda()
            self.model_params += list(self.inference_network.parameters())

        self.loss_fn = loss_fn['mse']
        if self.mode == 'meta-train':
            self.meta_optimizer = torch.optim.Adam(self.model_params, lr=self.meta_lr)
            self.scheduler = None

            # Set the logger
            set_logger(os.path.join(self.save_path, 'log.txt'))
            if args.use_wandb:
                wandb.init(entity="hayeonlee", 
                            project=args.project, 
                            name=args.exp_name, 
                            group=args.group, 
                            reinit=True)
                wandb.config.update(args)   
                writer = None     
            else:
                writer = SummaryWriter(log_dir=self.save_path)
            self.log = {
                        'meta_train': Log(self.save_path, 
                                            self.save_summary_steps, 
                                            self.metrics, 
                                            self.meta_train_devices, 
                                            'meta_train', 
                                            writer, args.use_wandb),
                        'meta_valid': Log(self.save_path, 
                                            self.save_summary_steps, 
                                            self.metrics, 
                                            self.meta_valid_devices, 
                                            'meta_valid', 
                                            writer, 
                                            args.use_wandb),
                        }


    def define_task_lr_params(self):
        self.task_lr = OrderedDict()
        for key, val in self.model.named_parameters():
            self.task_lr[key] = nn.Parameter(
                1e-3 * torch.ones_like(val))


    def get_params_z(self, xs, ys, hw_embed):
        params = self.model.cloned_params()

        z, kl = self.inference_network((xs, ys, hw_embed))
        zs = self.z_scaling
        for i, (name, weight) in enumerate(params.items()):
            if 'weight' in name:
                if 'fc3' in name:
                    idx = 0
                elif 'fc4' in name:
                    idx = 1
                elif 'fc5' in name:
                    idx = 2
                else:
                    continue
                layer_size = 2*self.layer_size
                params[name] = weight * (1 + zs*z['w'][idx*layer_size:(idx+1)*layer_size])

            elif 'bias' in name:
                if 'fc3' in name:
                    idx = 0
                elif 'fc4' in name:
                    idx = 1
                elif 'fc5' in name:
                    idx = 2
                else:
                    continue
                params[name] = weight + zs*z['b'][idx]
            else: raise ValueError(name)
        return params, kl, z 

    def train_single_task(self, hw_embed, xs, ys, num_updates):
        self.model.train()
        if self.search_space in ['fbnet', 'ofa']:
            xs, ys = xs.cuda(), ys.cuda()
        elif self.search_space == 'nasbench201':
            xs, ys = (xs[0].cuda(), xs[1].cuda()), ys.cuda() 
        hw_embed = hw_embed.cuda()
        if self.z_on:
            params, kl, z = self.get_params_z(xs, ys, hw_embed)
        else:
            params = self.model.cloned_params()
            kl = 0.0

        adapted_params = params
        
        for n in range(num_updates):
            ys_hat = self.model(xs, hw_embed, adapted_params)
            loss = self.loss_fn(ys_hat, ys)

            grads = torch.autograd.grad(
                loss, adapted_params.values(), create_graph=(self.second_order))

            for (key, val), grad in zip(adapted_params.items(), grads):
                if self.task_lr is not None: # Meta-SGD
                    task_lr = self.task_lr[key]
                else:
                    task_lr = self.inner_lr # MAML
                adapted_params[key] = val - task_lr * grad
        return adapted_params, kl


    def meta_train(self):
        print('==> start training...')
        max_valid_corr = -1

        if self.z_on:
            self.inference_network.train()

        with tqdm(total=self.num_episodes) as t:
            for i_epi in range(self.num_episodes):
                # Run inner loops to get adapted parameters (theta_t`)
                adapted_state_dicts = []
                query_list = []
                episode = self.data.generate_episode()
                for i_task in range(self.num_inner_tasks):
                    # Perform a gradient descent to meta-learner on the task
                    (hw_embed, xs, ys, xq, yq, _) = episode[i_task]

                    adapted_state_dict, kl_loss = \
                        self.train_single_task(hw_embed, xs, ys, self.num_train_updates)
                    # Store adapted parameters
                    # Store dataloaders for meta-update and evaluation
                    adapted_state_dicts.append(adapted_state_dict)
                    query_list.append((hw_embed, xq, yq))

                # Update the parameters of meta-learner
                # Compute losses with adapted parameters along with corresponding tasks
                # Updated the parameters of meta-learner using sum of the losses
                meta_loss = 0
                for i_task in range(self.num_inner_tasks):
                    hw_embed, xq, yq = query_list[i_task]
                    if self.search_space in ['fbnet', 'ofa']:
                        xq, yq = xq.cuda(), yq.cuda()
                    elif self.search_space == 'nasbench201':
                        xq, yq = (xq[0].cuda(), xq[1].cuda()), yq.cuda()
                    hw_embed = hw_embed.cuda()
                    adapted_state_dict = adapted_state_dicts[i_task]
                    yq_hat = self.model(xq, hw_embed, adapted_state_dict)
                    loss_t = self.loss_fn(yq_hat, yq)
                    meta_loss += loss_t / float(self.num_inner_tasks) \
                                            + self.kl_scaling * kl_loss

                self.meta_optimizer.zero_grad()
                meta_loss.backward()
                self.meta_optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step(meta_loss)

                # Evaluate model on new tasks
                # Evaluate on train and test dataset given a number of tasks (args.num_steps)                
                if (i_epi + 1) % self.save_summary_steps == 0:
                    logging.info(f"Episode {i_epi+1}/{self.num_episodes}")
                    postfix = {}
                    for split in ['meta_train', 'meta_valid']:
                        msg = f"[{split.upper()}] "
                        self._test_predictor(split, i_epi)
                        self.log[split].update_epi(i_epi)
                        for m in self.metrics + ['mse_loss', 'kl_loss']:
                            v = self.log[split].avg(i_epi, m)
                            postfix[f'{split}/{m}'] = f'{v:05.3f}'
                            msg += f"{m}: {v:05.3f}; " 

                            if m == 'spearman' and  max_valid_corr < v:
                                max_valid_corr = v 
                                save_dict = {'epi': i_epi,
                                             'model': self.model.cpu().state_dict()}
                                if self.args.z_on:
                                    save_dict['inference_network'] = self.inference_network.cpu().state_dict()
                                    self.inference_network.cuda()
                                if self.args.alpha_on:
                                    save_dict['task_lr'] = {k: v.cpu() for k, v in self.task_lr.items()}
                                    for k, v in self.task_lr.items():
                                       self.task_lr[k].cuda()
                                save_path = os.path.join(self.save_path, 'checkpoint', f'help_max_corr.pt')
                                torch.save(save_dict, save_path)
                                print(f'==> save {save_path}')
                                self.model.cuda()
                        logging.info(msg)
                    t.set_postfix(postfix)
                    print('\n')
                t.update()
        self.log['meta_train'].save()
        self.log['meta_valid'].save()
        print('==> Training done')


    def test_predictor(self):
        loaded = torch.load(self.load_path)
        print(f'==> load {self.load_path}')
        if 'epi' in loaded.keys():
            epi = loaded['epi']
            print(f'==> load {epi} model..')
        self.model.load_state_dict(loaded['model'])
        if self.z_on:
            self.inference_network.load_state_dict(loaded['inference_network'])
        if self.alpha_on:
            for (k, v), (lk, lv) in zip(self.task_lr.items(), loaded['task_lr'].items()):
                self.task_lr[k] = lv.cuda()
            
        self._test_predictor('meta_test', None)


    def _test_predictor(self, split, i_epi=None):
        save_file_path = os.path.join(self.save_path, f'test_log.txt')
        f = open(save_file_path, 'a+')

        if self.z_on:
            self.inference_network.eval()
        avg_metrics = {m: 0.0 for m in self.metrics}
        avg_metrics['mse_loss'] = 0.0

        tasks = self.data.generate_test_tasks(split) 
        for (hw_embed, xs, ys, xq, yq, device) in tasks:
            yq_hat_mean = None
            for _ in range(self.mc_sampling):
                adapted_state_dict, kl_loss = \
                    self.train_single_task(hw_embed, xs, ys, self.num_eval_updates)
                if self.search_space in ['fbnet', 'ofa']:
                    xq, yq = xq.cuda(), yq.cuda()
                elif self.search_space == 'nasbench201':
                    xq, yq = (xq[0].cuda(), xq[1].cuda()), yq.cuda()
                hw_embed = hw_embed.cuda()
                yq_hat = self.model(xq, hw_embed, adapted_state_dict)
                if yq_hat_mean is None:
                    yq_hat_mean = yq_hat
                else:
                    yq_hat_mean += yq_hat
            yq_hat_mean = yq_hat_mean / self.args.mc_sampling
            loss = self.loss_fn(yq_hat_mean, yq)  
            
            if i_epi is not None:
                for metric in self.metrics:
                    self.log[split].update(i_epi, metric, device, 
                                            val=metrics_fn[metric](yq_hat, yq)[0])
                self.log[split].update(i_epi, 'mse_loss', device, val=loss.item())
                self.log[split].update(i_epi, 'kl_loss', device, val=kl_loss if isinstance(kl_loss, float) else kl_loss.item())
            else:
                msg = f'[{split}/{device}] '
                for m in self.metrics:
                    msg += f'{m} {metrics_fn[m](yq_hat, yq)[0]:.3f} '
                    avg_metrics[m] += metrics_fn[m](yq_hat, yq)[0]
                msg += f'MSE {loss.item():.3f}'
                avg_metrics['mse_loss'] += loss.item()
                f.write(msg+'\n')
                print(msg)

        if i_epi is None:
            nd = len(tasks)
            msg = f'[{split}/average] '
            for m in self.metrics:
                msg += f'{m} {avg_metrics[m]/nd:.3f} '
            mse_loss = avg_metrics['mse_loss']
            msg += f'MSE {mse_loss/nd:.3f} ({nd} devices)'
            f.write(msg+'\n')
            print(msg)
        f.close()
    

    def _denormalization(self, task, yq_hat, adapted_state_dict):
        hw_embed, xs, ys, xq, yq, device, ys_gt, yq_gt = task
        xs =  (xs[0].cuda(), xs[1].cuda())
        ys_gt, yq_gt = ys_gt.cuda(), yq_gt.cuda()
        ys_hat = self.model(xs, hw_embed.cuda(), adapted_state_dict)
        ysh_min = min(ys_hat)
        ysh_max = max(ys_hat)

        denorm_yq_hat = denorm((yq_hat-ysh_min)/(ysh_max-ysh_min), max(ys_gt), min(ys_gt))
        denorm_mse = self.loss_fn(denorm_yq_hat.cuda(), yq_gt) 
        return denorm_yq_hat, denorm_mse 
      
    def load_model(self):
        loaded = torch.load(os.path.join(self.load_path))
        self.model.load_state_dict(loaded['model'])
        self.model.eval()
        self.model.cuda()
        if self.alpha_on:
            self.task_lr = {k: v.cuda() for k, v in loaded['task_lr'].items()}
        if self.z_on:
            self.inference_network.load_state_dict(loaded['inference_network'])
            self.inference_network.eval()
            self.inference_network.cuda()

    def nas(self):
        if self.search_space == 'ofa':
            self._nas_ofa()
        elif self.search_space == 'nasbench201':
            self._nas_metad2a()

    def _nas_metad2a(self):
        save_file_path = os.path.join(self.save_path, f'nas_results_{self.nas_target_device}.txt')
        f = open(save_file_path, 'a+')
        
        self.load_model()

        search_results = {}
        task = self.data.get_nas_task(self.nas_target_device)
        hw_embed, xs, ys, xq, yq, device, ys_gt, yq_gt = task

        yq_hat_mean = None
        for _ in range(self.mc_sampling):
            adapted_state_dict, kl_loss = \
                self.train_single_task(hw_embed, xs, ys, self.num_eval_updates)
            xq, yq = (xq[0].cuda(), xq[1].cuda()), yq.cuda()
            hw_embed = hw_embed.cuda()
            yq_hat = self.model(xq, hw_embed, adapted_state_dict)
            if yq_hat_mean is None:
                yq_hat_mean = yq_hat
            else:
                yq_hat_mean += yq_hat
        yq_hat_mean = yq_hat_mean / self.args.mc_sampling
        loss = self.loss_fn(yq_hat_mean, yq)  

        # Denormalization
        denorm_yq_hat, denorm_mse = self._denormalization(task, yq_hat_mean, adapted_state_dict)
        search_results = []
        top = 3
        true_acc = self.data.arch_candidates['true_acc']
        arch_str = self.data.arch_candidates['arch']
        const = float(self.latency_constraint)
        for dyq_hat, yq_, acc_, arch_ in \
                            zip(denorm_yq_hat, yq_gt, true_acc, arch_str):
            if dyq_hat.item() <= const:
                if len(search_results) < top:
                    search_results.append({
                        'yq': yq_, 
                        'acc': acc_, 
                        'arch_str': arch_
                    })

                if len(search_results) >= top:
                    break
        max_acc_result = search_results[0]
        for result in search_results:
            if result['acc'] > max_acc_result['acc']:
                max_acc_result = result
        lat = max_acc_result['yq'].item()
        acc = float(max_acc_result['acc'])
        arch = max_acc_result['arch_str']
        msg = f'[NAS Result] Target Device {self.nas_target_device} Constraint {const} '
        msg += f'| Latency {lat:.1f} | Accuracy {acc:.1f} | Neural Architecture {arch}'
        print(msg)
        f.write(msg+'\n')
        f.close()


    def _nas_ofa(self):
        from ofa.tutorial.accuracy_predictor import AccuracyPredictor
        from ofa.finder import EvolutionFinder
        
        # load HELP 
        self.load_model()

        task = self.data.get_nas_task(self.nas_target_device)
        #hw_embed, xs, ys, ys_gt = task
        #import pdb; pdb.set_trace()
        hw_embed, xs, ys, ys_gt = [_.cuda() for _ in task]
        ys_hat_mean = None
        for _ in range(self.mc_sampling):
            adapted_state_dict, kl_loss = \
                self.train_single_task(hw_embed, xs, ys, self.num_eval_updates)
            ys_hat = self.model(xs, hw_embed, adapted_state_dict)
            if ys_hat_mean is None:
                ys_hat_mean = ys_hat
            else:
                ys_hat_mean += ys_hat
        ys_hat = ys_hat_mean / self.args.mc_sampling

        latency_constraint = data_norm(self.latency_constraint, ys_gt, ys_hat).item()
        # load accuracy predictor of once-for-all
        acc_predictor = AccuracyPredictor(pretrained=True)
        params = {
            'constraint_type': self.nas_target_device, 
            'efficiency_constraint': latency_constraint,
            'hardware_embedding': hw_embed,
            'adapted_state_dict': adapted_state_dict,
            'mutate_prob': 0.1, # The probability of mutation in evolutionary search
            'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
            'efficiency_predictor':  self.model , # To use a predefined efficiency predictor.
            'accuracy_predictor': acc_predictor, # To use a predefined accuracy_predictor predictor.
            'ys_gt' : ys_gt,
            'ys_hat': ys_hat,
            'population_size': 100,
            'max_time_budget': 500,
            'parent_ratio': 0.25,
        }
        
        finder = EvolutionFinder(**params)
        best_valids, best_info, top_k = finder.run_evolution_search()
        pred_acc = best_info[0]
        arch_config = best_info[1]
        pred_lat = data_norm(best_info[2], ys_hat, ys_gt).item()

        msg = f'[NAS Result] Target Device {self.nas_target_device} '
        msg += f'Constraint {self.latency_constraint} '
        msg += f'Neural Architecture Config {arch_config}'
        print(msg)
        save_file_path = os.path.join(self.save_path, f'nas_results_{self.nas_target_device}.json')
        print(f'save path is {save_file_path}')
        json.dump(arch_config, open(save_file_path, 'w'), indent=4)
