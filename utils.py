# -*- coding: utf-8 -*-
import os
import json
import time
import logging
import torch
import pandas as pd
from itertools import product

def compute_subAcc_withlogits_binary(logits, target, a):
    # output is logits and predict_prob is probability
    assert logits.shape == target.shape, f"Acc, output {logits.shape} and target {target.shape} are not matched!"
    predict_prob = torch.sigmoid(logits)
    predict_prob = predict_prob.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    a = a.cpu().detach().numpy()
    # Young
    tmp = a <= 0
    predict_prob_n = predict_prob[tmp.nonzero()]
    target_n = target[tmp.nonzero()]
    Acc_n = (predict_prob_n.round() == target_n).mean()
    tmp = a > 0
    predict_prob_p = predict_prob[tmp.nonzero()]
    target_p = target[tmp.nonzero()]
    Acc_p = (predict_prob_p.round() == target_p).mean()
    return Acc_p, Acc_n


def run(command_template, qos, gpu, *args):
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    # if not os.path.exists('errors'):
    #     os.makedirs('errors')

    l = len(args)
    job_name_template = '{}'
    for _ in range(l-1):
        job_name_template += '-{}'
    for a in product(*args):
        command = command_template.format(*a)
        job_name = job_name_template.format(*a)
        bash_file = '{}.sh'.format(job_name)
        with open( bash_file, 'w' ) as OUT:
            OUT.write('#!/bin/bash\n')
            OUT.write('#SBATCH --job-name={} \n'.format(job_name))
            OUT.write('#SBATCH --ntasks=1 \n')
            OUT.write('#SBATCH --account=other \n')
            OUT.write(f'#SBATCH --qos={qos} \n')
            OUT.write('#SBATCH --partition=ALL \n')
            OUT.write('#SBATCH --cpus-per-task=4 \n')
            OUT.write(f'#SBATCH --gres=gpu:{gpu} \n')
            OUT.write('#SBATCH --mem={}G \n'.format(16 * gpu))
            OUT.write('#SBATCH --time=5-00:00:00 \n')
            OUT.write('#SBATCH --exclude=vista[03] \n')
            OUT.write('#SBATCH --output=outputs/{}.out \n'.format(job_name))
            OUT.write('#SBATCH --error=outputs/{}.out \n'.format(job_name))
            OUT.write('source ~/.bashrc\n')
            OUT.write('echo $HOSTNAME\n')
            OUT.write('echo "total gpu resources allocated: "$CUDA_VISIBLE_DEVICES\n')
            OUT.write('conda activate pytorch\n')
            OUT.write(command)
        qsub_command = 'sbatch {}'.format(bash_file)
        os.system( qsub_command )
        os.system('rm -f {}'.format(bash_file))
        print( qsub_command )
        print( 'Submitted' )

def append_data_to_csv(data,csv_name):
    df = pd.DataFrame(data)
    if os.path.exists(csv_name):
        df.to_csv(csv_name,mode='a',index=False,header=False)
    else:
        df.to_csv(csv_name,index=False)

def save_option(option):
    option_path = os.path.join(option.save_dir, option.exp_name, str(option.color_var), "options.json")

    with open(option_path, 'w') as fp:
        json.dump(option.__dict__, fp, indent=4, sort_keys=True)

def save_option_CelebA(option):
    option_path = os.path.join(option.save_dir, option.exp_name, option.attributes[0], option.eval_mode, "options.json")

    with open(option_path, 'w') as fp:
        json.dump(option.__dict__, fp, indent=4, sort_keys=True)

def save_option_IMDB(option):
    option_path = os.path.join(option.save_dir, option.exp_name, option.IMDB_train_mode, option.IMDB_test_mode, "options.json")

    with open(option_path, 'w') as fp:
        json.dump(option.__dict__, fp, indent=4, sort_keys=True)

def save_option_German(option):
    option_path = os.path.join(option.save_dir, option.exp_name, option.German_train_mode, option.German_test_mode, "options.json")

    with open(option_path, 'w') as fp:
        json.dump(option.__dict__, fp, indent=4, sort_keys=True)

def save_option_Diabetes(option):
    if option.bias_type == "I":
        option_path = os.path.join(option.save_dir, option.exp_name, option.minority, str(option.minority_size), "option.json")
    elif option.bias_type == "II":
        option_path = os.path.join(option.save_dir, option.exp_name, option.Diabetes_train_mode, option.Diabetes_test_mode, "option.json")

    with open(option_path, 'w') as fp:
        json.dump(option.__dict__, fp, indent=4, sort_keys=True)

def save_option_Adult(option):
    option_path = os.path.join(option.save_dir, option.exp_name, option.Adult_train_mode, option.Adult_test_mode, "options.json")

    with open(option_path, 'w') as fp:
        json.dump(option.__dict__, fp, indent=4, sort_keys=True)

def logger_setting(exp_name, save_dir, debug, filename='train.log'):
    logger = logging.getLogger(exp_name)
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')

    log_out = os.path.join(save_dir, filename)
    file_handler = logging.FileHandler(log_out)
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger


def printandlog(str1, savefilepath):
    print(str1)
    with open(os.path.join(savefilepath, 'log.txt'), 'a+') as f:
        f.write(str1)
        f.write('\n')


def _num_correct(outputs, labels, topk=1):
    _, preds = outputs.topk(k=topk, dim=1)
    preds = preds.t()
    correct = preds.eq(labels.view(1, -1).expand_as(preds))
    correct = correct.view(-1).sum()
    return correct

def _num_correct_CelebA(outputs, labels):
    preds = torch.sigmoid(outputs)
    # print(preds.size())
    # print(labels.size())
    # print('djsladjad')
    correct = (preds.round().view(-1) == labels.view(-1)).sum()
    # print(correct)
    return correct

def _accuracy( outputs, labels):
    batch_size = labels.size(0)
    _, preds = outputs.topk(k=1, dim=1)
    preds = preds.t()
    correct = preds.eq(labels.view(1, -1).expand_as(preds))
    correct = correct.view(-1).float().sum(0, keepdim=True)
    accuracy = correct.mul_(100.0 / batch_size)
    return accuracy

import pickle
def load_pkl(load_path):
    with open(load_path, "rb") as f:
        pkl_data = pickle.load(f)
    return pkl_data


def split_by_attr(key_list, target_dict, sex_dict):
    pp, pn, np, nn = [], [], [], []
    for k in key_list:
        if sex_dict[k] == 1 and target_dict[k] == 1:
            pp.append(k)
        elif sex_dict[k] == 1 and target_dict[k] == 0:
            pn.append(k)
        elif sex_dict[k] == 0 and target_dict[k] == 1:
            np.append(k)
        elif sex_dict[k] == 0 and target_dict[k] == 0:
            nn.append(k)
    return pp, pn, np, nn

def CelebA_eval_mode(key_list, target_dict, sex_dict, mode, train_or_test):
    pp, pn, np, nn = split_by_attr(key_list, target_dict, sex_dict)
    print(f'[{mode}] {train_or_test}: (Male-BlondHair) pp = {len(pp)}, pn = {len(pn)}, np = {len(np)}, nn = {len(nn)}')
    m = min(len(pp), len(pn), len(np), len(nn))
    if train_or_test == 'test' or train_or_test == 'dev':
        if mode.startswith('unbiased'):
            r_key_list = pp[:m] + pn[:m] + np[:m] + nn[:m]
        elif mode.startswith('conflict') and not mode.startswith('conflict_pp'):
            r_key_list = pp[:m] + nn[:m]
        elif mode.startswith('conflict_pp'):
            r_key_list = pp
    elif train_or_test == 'train':
        if mode == 'unbiased_ex' or mode == 'conflict_ex' or mode == 'conflict_pp_ex':
            r_key_list = pn + np
        else:
            r_key_list = key_list
    pp, pn, np, nn = split_by_attr(r_key_list, target_dict, sex_dict)
    print(f'[{mode}] {train_or_test}: (Male-BlondHair) pp = {len(pp)}, pn = {len(pn)}, np = {len(np)}, nn = {len(nn)}')
    return r_key_list

def get_attr_index(attributes):
    all_attr = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbone", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young", "Male"]
    res = []
    for attribute in attributes:
        res.append(all_attr.index(attribute))
    return res

def transfer_origin_for_testing_only(testing_dev_target_dict, attribute_list):
    res = dict()
    for k, v in testing_dev_target_dict.items():
        res[k] = v[attribute_list].reshape(len(attribute_list))
    return res

import re
def str_list(s):
    if type(s) is type([]):
        return s
    range_re = re.compile(r"^(\d+)-(\d+)$")
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(",")
    return [x for x in vals]