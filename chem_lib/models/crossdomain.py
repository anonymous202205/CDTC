# coding:utf-8
import random
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import auroc
from torch_geometric.data import DataLoader

from .maml import MAML
from ..datasets import sample_meta_datasets, sample_test_datasets, MoleculeDataset
from ..utils import Logger
import sys
from collections import OrderedDict
from class_dependency import *

from torch.utils.tensorboard import SummaryWriter

from ..scheduler import Scheduler

from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)

from learn2learn import detach_module

import pickle


def update_moving_avg(mv, reward, count):
    return mv + (reward.item() - mv) / (count + 1)


def get_inner_loop_parameter_dict(params, device):
    param_dict = dict()
    indexes = []
    for i, (name, param) in enumerate(params):
        if param.requires_grad:
            if "1.weight" in name or '1.bias' in name: continue
            param_dict[name] = param.to(device=device)
            indexes.append(i)

    return param_dict, indexes


class cd_maml(nn.Module):
    def __init__(self, args, model):
        super(cd_maml, self).__init__()
        self.args = args
        self.base_model = model
        self.model = MAML(model, lr=args.inner_lr, first_order=not args.second_order, anil=False, allow_unused=True)

        self.criterion = nn.CrossEntropyLoss().to(args.device)
        self.dataset = args.dataset
        self.test_dataset = args.test_dataset if args.test_dataset is not None else args.dataset
        self.train_dataset_dict = args.train_dataset_task_dict # real task name
        self.test_dataset_dictvalue = args.test_dataset_task # real task name
        self.data_dir = args.data_dir
        self.real_task_map = args.real_task_map

        self.n_shot_train = args.n_shot_train
        self.n_shot_test = args.n_shot_test
        self.n_query = args.n_query

        self.device = args.device

        self.emb_dim = args.emb_dim

        self.batch_task = args.batch_task

        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.inner_update_step = args.inner_update_step
        self.meta_lr = args.meta_lr
        self.inner_lr = args.inner_lr
        self.trial_path = args.trial_path

        self.d_names = args.d_names
        self.applyrl = args.applyrl

        self.naivedim = args.naivedim
        trial_name = args.d_names + '_' + str(self.test_dataset) + '_' + args.enc_gnn + \
                     '_k' + str(args.n_shot_train)+ '_q' +str(args.n_query) + '_lr' + str(args.meta_lr) + '_inner' + str(args.inner_lr) + \
              '_step' + str(args.update_step_test) + '_ep' + str(args.epochs)

        logger = Logger(self.trial_path + '/results.txt', title=trial_name)
        log_names = ['Epoch']

        log_names += ['AUC-' + str(t) for t in range(len(self.test_dataset_dictvalue))]

        log_names += ['AUC-Avg', 'AUC-Mid', 'AUC-Best']
        logger.set_names(log_names)
        self.logger = logger

        ms_preload_train_data = {}
        if args.preload_train_data:
            if self.dataset == 'all':
                for sname in self.train_dataset_dict:
                    preload_train_data = {}
                    for task in self.train_dataset_dict[sname]:  # real task name
                        real_task = self.real_task_map[sname][task]
                        dataset = MoleculeDataset(self.data_dir + sname + "/new/" + str(real_task + 1),
                                                  dataset=sname)
                        preload_train_data[task] = dataset
                    ms_preload_train_data[sname] = preload_train_data  # real task name

        preload_test_data = {}
        if args.preload_test_data:
            for task in self.test_dataset_dictvalue:
                real_task = self.real_task_map[self.test_dataset][task]
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(real_task + 1),
                                          dataset=self.test_dataset)
                preload_test_data[task] = dataset

        self.preload_train_data = ms_preload_train_data
        self.preload_test_data = preload_test_data

        if 'train' in self.dataset and args.support_valid:
            val_data_name = self.dataset.replace('train', 'valid')

            preload_val_data = {}
            for task in self.train_dataset_dict:
                real_task = self.real_task_map[val_data_name][task]
                dataset = MoleculeDataset(self.data_dir + val_data_name + "/new/" + str(real_task + 1),
                                          dataset=val_data_name)
                preload_val_data[task] = dataset
            self.preload_valid_data = preload_val_data

        self.step_gin1 = args.step_gin1
        self.step_agent = args.step_agent
        self.meta_epoch = 0
        self.rl_epoch = 0
        self.best_auc = 0
        self.res_logs = []
        self.args = args
        self.Tsmodel = self.targetsource()
        self.epochs = args.epochs
        self.n_query_test = args.n_query_test
        self.term = '_'.join(list(self.train_dataset_dict.keys())) + '_T' + self.test_dataset

        self.agentlr = args.agentlr
        # self.topk = args.d_topk

        self.buffer_size = 6
        self.rl_adv = args.rl_adv
        names_weights_copy, indexes = get_inner_loop_parameter_dict(self.model.named_parameters(), self.device)
        self.scheduler = Scheduler(self.naivedim, self.buffer_size, grad_indexes=indexes, pi=args.pi, device=self.device).to(self.device)

        self.scheduler_optimizer = torch.optim.Adam(self.scheduler.parameters(), lr=self.agentlr)

        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=args.meta_lr, weight_decay=args.weight_decay)

    def update_params(self, loss, model, update_lr):
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True , allow_unused=True)
        new_list  = []
        for i, g in enumerate(grads):
            if g is None:
                new_list.append([p for p in model.parameters()][i])
            else:
                n = [p for p in model.parameters()][i] - g * update_lr
                new_list.append(n)

        return grads, parameters_to_vector(new_list)

    def loader_to_samples(self, data):
        loader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=0)
        for samples in loader:
            samples = samples.to(self.device)
            return samples

    def get_data_sample(self, sname, task_id, train=True, t2t=False, test_source=False):
        real_task = self.real_task_map[sname][task_id]
        assert t2t is False
        assert test_source is False
        if train:
            if t2t is True:
                if task_id in self.preload_test_data:
                    dataset = self.preload_test_data[task_id]
            else:
                if self.dataset == 'all':
                    dataset = self.preload_train_data[sname][task_id]
                else:
                    dataset = MoleculeDataset(self.data_dir + self.dataset + "/new/" + str(real_task + 1), dataset=self.dataset)

            s_data, q_data = sample_meta_datasets(dataset, sname, task_id, real_task, self.n_shot_train, self.n_query)

            s_data = self.loader_to_samples(s_data)
            q_data = self.loader_to_samples(q_data)

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'q_label': q_data.y,
                            'label': torch.cat([s_data.y, q_data.y], 0)}
            eval_data = {}

        else:
            if task_id in self.preload_test_data:
                dataset = self.preload_test_data[task_id]
            else:
                dataset = MoleculeDataset(self.data_dir + self.test_dataset + "/new/" + str(real_task + 1),
                                          dataset=self.test_dataset)

            s_data, q_data, q_data_adapt = sample_test_datasets(dataset, self.test_dataset, task_id, real_task,
                                                                self.n_shot_test, self.n_query, self.n_query_test, self.update_step_test)
            s_data = self.loader_to_samples(s_data)
            q_loader = DataLoader(q_data, batch_size=self.n_query, shuffle=True, num_workers=0)
            q_loader_adapt = DataLoader(q_data_adapt, batch_size=self.n_query, shuffle=True, num_workers=0)

            adapt_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader_adapt}

            eval_data = {'s_data': s_data, 's_label': s_data.y, 'data_loader': q_loader}

        return adapt_data, eval_data

    def get_prediction(self, model, data, train=True):
        if train:
            s_logits, q_logits, adj, node_emb = model(data['s_data'], data['q_data'], data['s_label'])
            pred_dict = {'s_logits': s_logits, 'q_logits': q_logits, 'adj': adj, 'node_emb': node_emb}
        else:
            s_logits, logits,labels,adj_list,sup_labels = model.forward_query_loader(data['s_data'], data['data_loader'], data['s_label'])
            pred_dict = {'s_logits':s_logits, 'logits': logits, 'labels': labels,'adj':adj_list,'sup_labels':sup_labels}
        return pred_dict

    def get_adaptable_weights(self, model, adapt_mode):
        fenc = lambda x: x[0] == 'mol_encoder'
        fencls = lambda x: x[0] == 'cls'
        fenmlp = lambda x: x[0] == 'encode_projection'
        fenmod = lambda x: x[0] == 'mod'
        if adapt_mode == 'enc':
            flag = lambda x: fenc(x)
        elif adapt_mode == 'all':
            flag = lambda x: fenc(x) or fencls(x) or fenmlp(x) or fenmod(x)

        adaptable_names, adaptable_weights = [], []
        for name, p in model.module.named_parameters():
            names = name.split('.')

            if flag(names):
                adaptable_weights.append(p)
                adaptable_names.append(name)

        return adaptable_weights, adaptable_names

    def get_loss(self, model, batch_data, pred_dict, train=True, flag = 1):
        if not train:
            losses_adapt = self.criterion(pred_dict['s_logits'], batch_data['s_label'])
        else:
            if flag:
                losses_adapt = self.criterion(pred_dict['s_logits'], batch_data['s_label'])
            else:
                losses_adapt = self.criterion(pred_dict['q_logits'], batch_data['q_label'])

        if torch.isnan(losses_adapt).any() or torch.isinf(losses_adapt).any():
            losses_adapt = torch.zeros_like(losses_adapt)
        assert self.args.reg_adj == 0
        return losses_adapt

    def get_acc_reward(self, model, batch_data, pred_dict, train=True, flag = 0):
        if train:
            with torch.no_grad():
                if flag == 1:
                    _, predicted = torch.max(pred_dict['s_logits'].data, 1)
                    train_acc = torch.mean(torch.tensor([i.float() for i in predicted == batch_data['s_label']], device=predicted.device))
                else:
                    _, predicted = torch.max(pred_dict['q_logits'].data, 1)
                    train_acc = torch.mean(
                        torch.tensor([i.float() for i in predicted == batch_data['q_label']], device=predicted.device))
        return train_acc

    def targetsource(self):
        self.target_sample_index= self.sampleTargetSupport()
        return TargetEmb(self.args)

    def getproto(self, model):
        sourcepos = self.sampleSourceSupport()
        source_emb = self.get_domain_emb(sourcepos, model)
        target_emb = self.get_domain_emb(self.target_sample_index, model)
        targetemb = self.Tsmodel(target_emb, self.test_dataset_dictvalue)
        sourceemb = self.Tsmodel(source_emb, self.train_dataset_dict)
        return sourceemb, targetemb

    def sampleSourceSupport(self):
        sample_id = OrderedDict()
        for sname in OrderedDict(self.train_dataset_dict):
            for cls in range(len(self.train_dataset_dict[sname])):
                db = self.get_pos_sample(sname, cls, train=True)
                key = (sname, cls)
                sample_id[key] = db
        return sample_id

    def sampleTargetSupport(self):
        sample_id = OrderedDict()
        sname = self.test_dataset
        for cls in range(len(self.test_dataset_dictvalue)):
            db = self.get_pos_sample(sname, cls, train=False)
            key = (sname, cls)
            sample_id[key] = db
        return sample_id

    def get_pos_sample(self, sname, task_id, train=True):
        real_task = self.real_task_map[sname][task_id]
        if not train:
            if task_id in self.preload_test_data:
                dataset = self.preload_test_data[task_id]
        else:
            dataset = self.preload_train_data[sname][task_id]

        s_data, q_data = sample_meta_datasets(dataset, sname, task_id, real_task, self.n_shot_train, self.n_query)

        s_data = self.loader_to_samples(s_data)
        q_data = self.loader_to_samples(q_data)
        adapt_data = {'s_data': s_data, 's_label': s_data.y, 'q_data': q_data, 'q_label': q_data.y,
                        'label': torch.cat([s_data.y, q_data.y], 0)}
        return adapt_data

    def get_domain_emb(self, domainb, model):
        emb_dict = OrderedDict()
        for sname_task in domainb:
            db = domainb[sname_task]
            s_data = db['s_data']
            q_data = db['q_data']
            s_emb, s_node_emb = model.mol_encoder(s_data.x, s_data.edge_index, s_data.edge_attr, s_data.batch)
            q_emb, q_node_emb = model.mol_encoder(q_data.x, q_data.edge_index, q_data.edge_attr, q_data.batch)
            s_emb, _ = model.encode_projection(s_emb, q_emb)
            emb_dict[sname_task] = s_emb
        return emb_dict

    def rl_step(self, writer, bef_model, loss_to_reduce, moving_avg, selected_tasks_idx,  step, rl_step):
        if self.applyrl:
            aft_model = bef_model.clone()
            detach_module(aft_model, keep_requires_grad=True)

            new_grad, new_params = self.update_params(loss_to_reduce, bef_model, update_lr=rl_step)
            vector_to_parameters(new_params, aft_model.parameters())

            with torch.no_grad():
                target_tasks = list(self.target_sample_index.keys())
                target_task_sel = target_tasks
                ac_target_bef = []
                loss_target_bef = []
                for task_sel in target_task_sel:
                    train_data = self.target_sample_index[task_sel]
                    inner_model_bef = bef_model.clone()
                    inner_model_bef.train()
                    pred_eval = self.get_prediction(inner_model_bef, train_data, train=True)  # train just support data
                    loss_eval = self.get_loss(inner_model_bef, train_data, pred_eval, train=True,
                                              flag=1)
                    ac_eval = self.get_acc_reward(inner_model_bef, train_data, pred_eval, train=True, flag=1)
                    ac_target_bef.append(ac_eval)
                    loss_target_bef.append(loss_eval)

                loss_target = []
                ac_target = []
                for task_sel in target_task_sel:
                    train_data = self.target_sample_index[task_sel]
                    inner_model = aft_model.clone()
                    inner_model.train()
                    pred_eval = self.get_prediction(inner_model, train_data, train=True)  # train just support data
                    loss_eval = self.get_loss(inner_model, train_data, pred_eval, train=True,
                                              flag=1)  # loss for support
                    ac_eval = self.get_acc_reward(inner_model, train_data, pred_eval, train=True, flag=1)
                    ac_target.append(ac_eval)
                    loss_target.append(loss_eval)

            reward_aft = torch.stack(loss_target).mean()
            reward_bef = torch.stack(loss_target_bef).mean()
            reward = (reward_bef - reward_aft).detach()

        probs = torch.tensor(0)
        for i in selected_tasks_idx:
            probs = probs - self.scheduler.m.log_prob(i)
        loss = probs * (reward - moving_avg)
        moving_avg_reward = update_moving_avg(moving_avg, reward, step)

        self.scheduler_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.scheduler.parameters(), 1)
        self.scheduler_optimizer.step()
        return moving_avg_reward

    def train_step(self, writer, step, moving_avg_reward):
        def sampleid2task(ids, dicttask):
            taskid = list(dicttask.keys())
            ids_name = []
            for i in ids:
                ids_name.append(taskid[i])
            return ids_name

        step_agent = self.step_agent
        trainpos = 'train1'
        step_gin = self.epochs
        step_gin1 = self.step_gin1
        rl = True
        if rl:
            sourceemb, targetemb = self.getproto(self.model)
            if self.applyrl:
                _, _, all_task_weight = self.scheduler.get_weight(sourceemb, targetemb)

            all_task_prob = all_task_weight.view(-1)
            selected_tasks_idx = self.scheduler.sample_task(all_task_prob, self.batch_task, replace=False)

            selected_tasks_idname = sampleid2task(selected_tasks_idx, sourceemb)

            data_batches = []
            for item in selected_tasks_idname:
                sname, task_id = item
                db = self.get_data_sample(sname, task_id, train=True, t2t=False)
                data_batches.append(db)

        self.update_step = 1
        for _ in range(self.update_step):
            losses_eval = []
            for task_id in range(self.batch_task):
                train_data, _ = data_batches[task_id]
                model = self.model.clone()
                model.train()
                adaptable_weights, adaptable_names = self.get_adaptable_weights(model, adapt_mode='all')

                for _ in range(self.inner_update_step):
                    pred_adapt = self.get_prediction(model, train_data, train=True)
                    loss_adapt = self.get_loss(model, train_data, pred_adapt, train=True, flag=1)
                    model.adapt(loss_adapt,  adaptable_weights=adaptable_weights)
                pred_eval = self.get_prediction(model, train_data, train=True)
                loss_eval = self.get_loss(model, train_data, pred_eval, train=True, flag=0)
                losses_eval.append(loss_eval)

            losses_eval = torch.stack(losses_eval)

            losses_eval = torch.sum(losses_eval)

            losses_eval = losses_eval / self.batch_task

            if self.applyrl:

                if trainpos == 'train1' and step > step_gin1:
                    moving_avg_reward = self.rl_step(writer, self.model, losses_eval, moving_avg_reward,
                                             selected_tasks_idx, step, self.rl_adv)

                self.rl_epoch += 1

        if step <= step_gin1 or (step > step_agent) & (step <= step_gin):
            self.meta_epoch += 1
            self.optimizer.zero_grad()
            losses_eval.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

        return self.model.module, moving_avg_reward

    def test_step(self, writer, step):
        step_results = {'query_preds': [], 'query_labels': [], 'query_adj': [], 'task_index': []}
        taskpool = self.test_dataset_dictvalue
        taskpool = range(len(taskpool))
        ttsplit = taskpool

        auc_scores = []
        for task_id in ttsplit:
            adapt_data, eval_data = self.get_data_sample(self.test_dataset, task_id, train=False)
            model = self.model.clone()
            model.train()
            for i in range(self.update_step_test):
                cur_adapt_data = {'s_data': adapt_data['s_data'], 's_label': adapt_data['s_label'],
                                  'q_data': adapt_data['s_data'], 'q_label': None}
                adaptable_weights, _ = self.get_adaptable_weights(model, adapt_mode='all')
                pred_adapt = self.get_prediction(model, cur_adapt_data, train=True)
                loss_adapt = self.get_loss(model, cur_adapt_data, pred_adapt, train=True, flag=1)

                model.adapt(loss_adapt, allow_nograd=False, adaptable_weights=adaptable_weights)

            model.eval()
            with torch.no_grad():
                pred_eval = self.get_prediction(model, eval_data, train=False)
                y_score = F.softmax(pred_eval['logits'], dim=-1).detach()[:, 1]
                y_true = pred_eval['labels']

                auc = auroc(y_score, y_true, pos_label=1).item()

            auc_scores.append(auc)
            if self.args.save_logs:
                step_results['query_preds'].append(y_score.cpu().numpy())
                step_results['query_labels'].append(y_true.cpu().numpy())
                step_results['query_adj'].append(pred_eval['adj'].cpu().numpy())
                step_results['task_index'].append(self.test_dataset_dictvalue[task_id])

        mid_auc = np.median(auc_scores)
        avg_auc = np.mean(auc_scores)
        self.best_auc = max(self.best_auc, avg_auc)  # compared to historical avg, get the best avg
        # self.logger.append([step] + auc_scores + [avg_auc, mid_auc, self.best_auc], verbose=False)
        print('Test Epoch:', step, 'from ',  '_'.join(list(self.train_dataset_dict.keys())),  'to ',
              self.test_dataset,  ', AUC_Mid:', round(mid_auc, 4), ', AUC_Avg: ', round(avg_auc, 4),
              ', Best_Avg_AUC: ', round(self.best_auc, 4), )
        if self.args.save_logs:
            self.res_logs.append(step_results)
        return self.best_auc

    def save_model(self, step):
        save_path = os.path.join(self.trial_path, f"step_{step}.pth")
        torch.save(self.model.module.state_dict(), save_path)
        print(f"Checkpoint saved in {save_path}")

    def save_result_log(self):
        joblib.dump(self.res_logs,self.args.trial_path+'/logs.pkl',compress=6)

    def conclude(self):
        df = self.logger.conclude()
        self.logger.close()

    def reinit_weights(self, m):
        if isinstance(m, nn.Module):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
