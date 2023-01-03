#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Chameleon implementation in PyTorch"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from hashlib import sha3_224

from data_loader_exp3 import CORE50
import copy
import os
import json
from models.mobilenet import MyMobilenetV1
from utils import *
import configparser
import argparse
from pprint import pprint
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import timeit
import heapq
import torchvision.models as models
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import torchvision
import torch.nn.functional as F
import numpy as np

# --------------------------------- Setup --------------------------------------

# recover exp configuration name
parser = argparse.ArgumentParser(description='Run CL experiments')
parser.add_argument('--name', dest='exp_name',  default='DEFAULT',
                    help='name of the experiment you want to run.')
parser.add_argument('--scenario', type=str, default="ni",
                    choices=['ni', 'nc', 'nic', 'nicv2_79', 'nicv2_196',
                                'nicv2_391'])
parser.add_argument('--save_dir', type=str, default="results",
                    help='directory to save experimental results')
parser.add_argument('--run', type=int, default=1,
                    help='directory to save experimental results')
args = parser.parse_args()


# directory for saving experimental results
args.save_dir = os.path.join(args.save_dir, args.scenario)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

replay_save_dir = os.path.join(args.save_dir, "replay_storage")
lt_replay_save_dir = os.path.join(args.save_dir, "lt_replay_storage")
if not os.path.exists(replay_save_dir):
    os.makedirs(replay_save_dir)

if not os.path.exists(lt_replay_save_dir):
    os.makedirs(lt_replay_save_dir)

# set cuda device (based on your hardware)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# recover config file for the experiment
config = configparser.ConfigParser()
config.read("params.cfg")
exp_config = config[args.exp_name]
print("Experiment name:", args.exp_name)
pprint(dict(exp_config))

# recover parameters from the cfg file and compute the dependent ones.
exp_name = eval(exp_config['exp_name'])
use_cuda = eval(exp_config['use_cuda'])
init_lr = eval(exp_config['init_lr'])
inc_lr = eval(exp_config['inc_lr'])
mb_size = eval(exp_config['mb_size'])
init_train_ep = eval(exp_config['init_train_ep'])
inc_train_ep = eval(exp_config['inc_train_ep'])
init_update_rate = eval(exp_config['init_update_rate'])
inc_update_rate = eval(exp_config['inc_update_rate'])
max_r_max = eval(exp_config['max_r_max'])
max_d_max = eval(exp_config['max_d_max'])
inc_step = eval(exp_config['inc_step'])
rm_sz = eval(exp_config['rm_sz'])
momentum = eval(exp_config['momentum'])
l2 = eval(exp_config['l2'])
freeze_below_layer = eval(exp_config['freeze_below_layer'])
latent_layer_num = eval(exp_config['latent_layer_num'])
reg_lambda = eval(exp_config['reg_lambda'])

# setting up log dir for tensorboard
log_dir = 'logs/' + exp_name
writer = SummaryWriter(log_dir)

# Saving params
hyper = json.dumps(dict(exp_config))
writer.add_text("parameters", hyper, 0)
total_samples_seen = 0
# Other variables init
tot_it_step = 0
rm = None
ltm = None
user_pref_list = [
'[21 46 5 7 43]',
'[47 27 23 9 41]', 
'[38 37 11 40 23]',
'[18 37 16 34 36]',
'[44 6 40 27 31]',
'[16 22 11 40 24]',
'[11 15 27 29 45]',
'[27 49 31 22 13]',
'[34 47 16 45 30]',
'[9 46 33 27 21]']
user_idx = int(args.run) - 1
user_pref_cls = list(map(int, user_pref_list[user_idx][1:-1].split(' ')))

prototypes = None
running_freq = {i:0 for i in range(50)}
replay_samples_st = {i:0 for i in range(50)}
replay_samples_lt = {i:0 for i in range(50)}
# Create the dataset object
dataset = CORE50(root='/home/shivam/NUS/QE/continual_learning/ar1-pytorch/dataset/core50_128x128', scenario=args.scenario, cumul=False,user_pref_cls=user_pref_cls)
preproc = preprocess_imgs

# Get the fixed test set
test_x, test_y = dataset.get_test_set()
ltm_cur_sz = 0 

#Size of Long-Term Store
lm_sz = 100 
lm_sz_cur = 0

# Model setup
model = MyMobilenetV1(pretrained=True, latent_layer_num=latent_layer_num)
# # we replace BN layers with Batch Renormalization layers
replace_bn_with_brn(
    model, momentum=init_update_rate, r_d_max_inc_step=inc_step,
    max_r_max=max_r_max, max_d_max=max_d_max
)
model.saved_weights = {}
model.past_j = {i:0 for i in range(50)}
model.cur_j = {i:0 for i in range(50)}

# Optimizer setup
optimizer = torch.optim.SGD(
    model.parameters(), lr=init_lr, momentum=momentum, weight_decay=l2
)
criterion2 = nn.CrossEntropyLoss()

setting = 'stream'

count_k = 0
count_n_k = 0
prob_k = 0.5
prob_n_k = 0.5
prob_class = list()

for cls in range(50):
    if cls in user_pref_cls:
        prob_class.append(prob_k)
    else:
        prob_class.append(prob_n_k)

prob_k_lm = 0.5
prob_n_k_lm = 0.5
prob_class_lm = list()
for cls in range(50):
    if cls in user_pref_cls:
        prob_class_lm.append(prob_k_lm)
    else:
        prob_class_lm.append(prob_n_k_lm)

avg_K = list()
avg_acc = list()
flag = -1
cls_num_list = [50] * 50
# --------------------------------- Training -----------------------------------

start_time = timeit.default_timer()

user_pref_cls_acc = {}

# loop over the training incremental batches
for i, train_batch in enumerate(dataset):

    print("Probability:", prob_k, prob_n_k)
    if reg_lambda != 0:
        init_batch(model, ewcData, synData)

    # we freeze the layer below the replay layer since the first batch
    freeze_up_to(model, freeze_below_layer, only_conv=False)

    if i == 1:
        change_brn_pars(
            model, momentum=inc_update_rate, r_d_max_inc_step=0,
            r_max=max_r_max, d_max=max_d_max)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=inc_lr, momentum=momentum, weight_decay=l2
        )

    train_x, train_y = train_batch
    train_x = preproc(train_x)

    if i == 0:
        cur_class = [int(o) for o in set(train_y)]
        model.cur_j = examples_per_class(train_y)
    else:
        cur_class = [int(o) for o in set(train_y).union(set(rm[1]))]
        model.cur_j = examples_per_class(list(train_y) + list(rm[1]))

    s1_size = 10
    print("----------- batch {0} -------------".format(i))
    print("train_x shape: {}, train_y shape: {}"
          .format(train_x.shape, train_y.shape))
    
    replay_samples_st = {i:0 for i in range(50)}
    replay_samples_lt = {i:0 for i in range(50)}
    model.lat_features.eval()
    model.end_features.train()
    start_rm = 0
    end_rm = 9
    start_lm = 0
    end_lm = 9

    reset_weights(model, cur_class)
    cur_ep = 0

    if i == 0:
        (train_x, train_y), it_x_ep = pad_data([train_x, train_y], mb_size)
    shuffle_in_unison([train_x, train_y], in_place=True)

    model = maybe_cuda(model, use_cuda=use_cuda)
    acc = None
    ave_loss = 0

    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    if i == 0:
        train_ep = init_train_ep
    else:
        train_ep = inc_train_ep

    for ep in range(train_ep):

        print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0

        # computing how many patterns to inject in the latent replay layer
        if i > 0:
            cur_sz = 1
            it_x_ep = train_x.size(0) // cur_sz
        else:
            n2inject = 0
        n2ltm = 0
        print("total sz:", train_x.size(0) + rm_sz)
        print("n2inject", n2inject)
        print("it x ep: ", it_x_ep)
        update_mem_cnt = 0
        for it in range(it_x_ep):

            if i > 0:
                if it % 10 == 0:
                    n2inject = 10
                    n2ltm = 1
                else:
                    n2inject = 10
                    n2ltm = 0

            if i > 0:
                mb_size = 1
            else:
                mb_size = mb_size

            start = it * (mb_size)
            end = (it + 1) * (mb_size)

            optimizer.zero_grad()

            x_mb = maybe_cuda(train_x[start:end], use_cuda=use_cuda)

            if i == 0:
                lat_mb_x = None
                y_mb = maybe_cuda(train_y[start:end], use_cuda=use_cuda)
                for sample_idx in range(start, end):
                    if train_y[sample_idx].item() in user_pref_cls:
                        count_k+=1
                    else:
                        count_n_k+=1
            else:
                running_freq[train_y[start:end].item()] += 1
                if train_y[start:end].item() in user_pref_cls:
                    count_k+=1
                else:
                    count_n_k+=1

                #select replay patterns to play at random
                if n2inject > 0:
                    replay_indx = [indx for indx in range(start_rm, end_rm + 1)]
                    start_rm = (start_rm + 10) % rm_sz
                    end_rm = (end_rm + 10) % rm_sz
                else:
                    replay_indx = []
                if n2ltm > 0:
                    ltm_replay_indx = [indx for indx in range(start_lm, end_lm + 1)]
                    start_lm = (start_lm + 10) % lm_sz
                    end_lm = (end_lm + 10) % lm_sz
                else:
                    ltm_replay_indx = []

                if len(replay_indx) > 0 and flag != -1:
                    lat_mb_x = [None] * (len(replay_indx) + len(ltm_replay_indx))
                    lat_mb_y = [None] * (len(replay_indx) + len(ltm_replay_indx))
                    for j, idx in enumerate(replay_indx):
                        lat_mb_x[j] = rm[0][idx]
                        lat_mb_y[j] = rm[1][idx]
                        replay_samples_st[lat_mb_y[j].item()] += 1
                    for k, idx in enumerate(ltm_replay_indx):
                        lat_mb_x[j + 1 + k] = ltm[0][idx]
                        lat_mb_y[j + 1 + k] = ltm[1][idx]
                        replay_samples_lt[lat_mb_y[j + 1 + k].item()] += 1
                    lat_mb_x = torch.stack(lat_mb_x)
                    lat_mb_y = torch.tensor(lat_mb_y).type(torch.LongTensor)
                else:
                    lat_mb_x = rm[0][it*n2inject: (it + 1)*n2inject]
                    lat_mb_y = rm[1][it*n2inject: (it + 1)*n2inject]
                y_mb = maybe_cuda(
                    torch.cat((train_y[start:end], lat_mb_y), 0),
                    use_cuda=use_cuda)
                lat_mb_x = maybe_cuda(lat_mb_x, use_cuda=use_cuda)

            # if lat_mb_x is not None, this tensor will be concatenated in
            # the forward pass on-the-fly in the latent replay layer
            logits, lat_acts, feat = model(
                x_mb, latent_input=lat_mb_x, return_lat_acts=True)

            prob_cur_acts = list()
            #adding samples in the short-term memory
            if i > 0 and it > 1 and it % s1_size == 0  and flag != -1:
                running_sum = 0
                for sample in cur_labels:
                    running_sum += prob_class[sample.item()]
                    cnt += 1
                for sample in cur_labels:
                    prob_cur_acts.append(prob_class[sample.item()]/running_sum)

                update_mem_cnt += 1
                replace_sz = rm_sz // (i + 1)
                h = min(replace_sz, cur_acts.size(0))

                outputs_stm = cur_logits
                labels_stm = cur_labels
                vals, losses = compute_replay_probabilities("logit_dist", outputs_stm, labels_stm.cuda(), criterion2, start_time=None)

                res, ind = torch.topk(vals.detach(), h, largest=False)
                ind = ind.tolist()
                idxs_cur = ind

                rm_add = [cur_acts[idxs_cur], cur_labels[idxs_cur]]

                #assigning prob to short-term samples
                prob_cur_acts_rm = list()
                running_sum = 0
                for sample in rm[1]:
                    running_sum += (1 - prob_class[sample.item()])
                for sample in rm[1]:
                    prob_cur_acts_rm.append((1 - prob_class[sample.item()])/running_sum)

                idxs_2_replace = np.random.choice(
                    rm[0].size(0), h, replace=False, p=prob_cur_acts_rm 
                )

                for j, idx in enumerate(idxs_2_replace):
                    rm[0][idx] = copy.deepcopy(rm_add[0][j])
                    rm[1][idx] = copy.deepcopy(rm_add[1][j])


            # collect latent volumes only for the first ep
            # we need to store them to eventually add them into the external
            # replay memory
            if ep == 0:
                lat_acts = lat_acts.cpu().detach()
                if i == 0:
                    if it == 0:
                        cur_acts = copy.deepcopy(lat_acts)
                        cur_labels = copy.deepcopy(train_y[start:end])
                        cur_logits = copy.deepcopy(logits[start:end].detach())
                    else:
                        cur_acts = torch.cat((cur_acts, lat_acts), 0)
                        cur_labels = torch.cat((cur_labels, train_y[start:end]), 0)
                        cur_logits = torch.cat((cur_logits, logits[start:end].detach()), 0)
                if i > 0:
                    if it % s1_size == 0:
                        cur_acts = copy.deepcopy(lat_acts)
                        cur_labels = copy.deepcopy(train_y[start:end])
                        cur_logits = copy.deepcopy(logits[0:1].detach())
                    else:
                        cur_acts = torch.cat((cur_acts, lat_acts), 0)
                        cur_labels = torch.cat((cur_labels, train_y[start:end]), 0)
                        cur_logits = torch.cat((cur_logits, logits[0:1].detach()), 0)

            _, pred_label = torch.max(logits, 1)
            correct_cnt += (pred_label == y_mb).sum()

            loss_all = criterion2(logits, y_mb)
            loss = loss_all

            #adding samples in the long-term memory
            prob_cur_acts = list()
            if i > 0 and it > 1 and it % s1_size == 0 and flag != -1 and n2ltm != 0:
                age_cur_acts = list()
                running_sum = 0
                for sample in rm[1]:
                    running_sum += prob_class_lm[sample.item()]
                    cnt += 1
                for sample in rm[1]:
                    prob_cur_acts.append(prob_class_lm[sample.item()]/running_sum)

                h = min(1, rm[0].size(0))
                outputs_rm = logits[1:11]
                labels_rm = y_mb[1:11]

                vals, losses = compute_replay_probabilities("logit_dist", outputs_rm, labels_rm, criterion2, start_time=None)

                ### Compute new class prototypes ###
                if prototypes is None:
                    prototypes = {}
                    num_prototype_samples = {}
                    for cls in range(50):
                        num_prototype_samples[cls] = 0
                    for ltm_sample_idx in range(ltm[0].size(0)):
                        if ltm[1][ltm_sample_idx].item() in prototypes.keys():
                            prototypes[ltm[1][ltm_sample_idx].item()] += ltm[0][ltm_sample_idx]
                        else:
                            prototypes[ltm[1][ltm_sample_idx].item()] = ltm[0][ltm_sample_idx]
                        num_prototype_samples[ltm[1][ltm_sample_idx].item()] += 1

                idxs_cur_gen = torch.argmin(vals.detach())
                idxs_cur_gen = [idxs_cur_gen.tolist()]
                idxs_cur_spec = [0]

                ### add information gain score here  ###
                if prototypes is not None:
                    prototypes_copy = prototypes
                    for pref_cls in prototypes.keys():
                        prototypes_copy[pref_cls] /= num_prototype_samples[pref_cls]
                    proto_lat_acts =  torch.stack(list(prototypes_copy.values())).cuda()
                    logits_cp = model(None, latent_input=proto_lat_acts, return_lat_acts=False, lat=False)
                    softmax = torch.nn.Softmax(dim=1)
                    _, pred_label_cl = torch.max(logits_cp, 1)

                    info_gain_scores = []
                    sample_pred_probs = softmax(logits[1:11])
                    proto_pred_probs =  softmax(logits_cp)

                    for sample_idx, sample_prob in enumerate(sample_pred_probs):
                        if y_mb[sample_idx] in prototypes.keys():
                            score = F.kl_div(sample_prob.log(), proto_pred_probs[list(prototypes.keys()).index(y_mb[sample_idx])], size_average=False)
                            score_prob =score
                            info_gain_scores.append(score_prob)

                    if info_gain_scores != []:
                        idxs_cur_spec = info_gain_scores.index(max(info_gain_scores))
                        idxs_cur_spec = [idxs_cur_spec]

                import random
                idxs_cur = idxs_cur_spec
                ltm_add = [rm[0][idxs_cur], rm[1][idxs_cur]]

                # # replace patterns in random memory
                per_class_sz = lm_sz // 50
                class_id = rm[1][idxs_cur]
                idx_to_choose_from  = np.arange(class_id*per_class_sz, (class_id + 1)*per_class_sz)
                idxs_2_replace = np.random.choice(
                    idx_to_choose_from, h, replace=False
                )
                for j, idx in enumerate(idxs_2_replace):
                    ltm[0][idx] = copy.deepcopy(ltm_add[0][j])
                    ltm[1][idx] = copy.deepcopy(ltm_add[1][j])
                
                if prototypes is not None:
                    iddd = list(idxs_2_replace)
                    for ltm_sample_idx in range(ltm[0].size(0)):
                        if ltm_sample_idx == iddd[0]:
                            if ltm[1][ltm_sample_idx].item() not in prototypes.keys():
                                prototypes[ltm[1][ltm_sample_idx].item()] = ltm[0][ltm_sample_idx]
                                num_prototype_samples[ltm[1][ltm_sample_idx].item()] = 1
                            else:                            
                                prototypes[ltm[1][ltm_sample_idx].item()] += ltm[0][ltm_sample_idx]
                                num_prototype_samples[ltm[1][ltm_sample_idx].item()] += 1
                

            ave_loss += loss.item()

            loss.backward()
            optimizer.step()

            acc = correct_cnt.item() / \
                  ((it + 1) * y_mb.size(0))
            ave_loss /= ((it + 1) * y_mb.size(0))

            if it % 10 == 0:
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}'
                        .format(it, ave_loss, acc)
                )

            # Log scalar values (scalar summary) to TB
            tot_it_step +=1
            writer.add_scalar('train_loss', ave_loss, tot_it_step)
            writer.add_scalar('train_accuracy', acc, tot_it_step)

            if i > 0  and sum(running_freq.values()) % 5000 == 0:
                flag = 1
                largest_cls = {key: running_freq[key] for key in sorted(running_freq, key=running_freq.get, reverse=True)[:5]}
                k_keys_sorted = list(largest_cls.keys())
                sum_k_classes = 0
                sum_n_k_classes = 0

                for class_id in  k_keys_sorted:
                    sum_k_classes += running_freq[class_id]
                total_sum = sum(running_freq.values())      
                sum_n_k_classes = (total_sum - sum_k_classes) / 45.0
                sum_k_classes = sum_k_classes / 5.0
                total_sum = sum_k_classes + sum_n_k_classes
                prob_k = (sum_k_classes / total_sum)

                prob_k = prob_k * 1.2
                prob_n_k = 1 - prob_k

                for cls in range(50):
                    if cls in  k_keys_sorted:
                        prob_class[cls] = prob_k
                    else:
                        prob_class[cls] = prob_n_k
                cls_num_list = list(running_freq.values())
            
        cur_ep += 1

    consolidate_weights(model, cur_class)
    
    print("Before rm", cur_acts.size(0))
    if i == 0:
        prob_cur_acts = list()
        running_sum = 0
        cnt = 0
        for sample in train_y:
            running_sum += prob_class[sample.item()]
            cnt += 1
        for sample in train_y:
            prob_cur_acts.append(prob_class[sample.item()]/running_sum)
        h = min(rm_sz // (i + 1), cur_acts.size(0))
        print("h", h)

        print("cur_acts sz:", cur_acts.size(0))
        idxs_cur = np.random.choice(
            cur_acts.size(0), h, replace=False
        )
        rm_add = [cur_acts[idxs_cur], train_y[idxs_cur]]
        print("rm_add size", rm_add[0].size(0))

        # replace patterns in random memory
        if i == 0:
            rm = copy.deepcopy(rm_add)
        else:
            idxs_2_replace = np.random.choice(
                rm[0].size(0), h, replace=False
            )
            for j, idx in enumerate(idxs_2_replace):
                rm[0][idx] = copy.deepcopy(rm_add[0][j])
                rm[1][idx] = copy.deepcopy(rm_add[1][j])

    
        idxs_lt = np.random.choice(
            cur_acts.size(0), lm_sz, replace=False
        )
        ltm_add = [cur_acts[idxs_lt], cur_labels[idxs_lt]]
        ltm = copy.deepcopy(ltm_add)

    set_consolidate_weights(model)

    ave_loss, acc, accs = get_accuracy(
        model, criterion2, 128, test_x, test_y, preproc=preproc
    )
    avg_K_acc = 0
    for cls in user_pref_cls:
        if i == 0:
            user_pref_cls_acc[cls] = [accs[cls]]    
        else:
            user_pref_cls_acc[cls].append(accs[cls])
        avg_K_acc += accs[cls]
        print("Accuracy for class " + str(cls) + ": ", accs[cls])
    avg_K_acc = avg_K_acc/5
    avg_K.append(avg_K_acc)
    avg_acc.append(acc)

    # Log scalar values (scalar summary) to TB
    writer.add_scalar('test_loss', ave_loss, i)
    writer.add_scalar('test_accuracy', acc, i)

    # update number examples encountered over time
    for c, n in model.cur_j.items():
        model.past_j[c] += n
            
    print("---------------------------------")
    print("Accuracy: ", acc)
    print("Average accuracy over K classes:", avg_K_acc)
    print("---------------------------------")

writer.close()
