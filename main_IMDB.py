# -*- coding: utf-8 -*-
import torch
from torch.backends import cudnn

import os
import random
from trainer_IMDB import Trainer
from utils import save_option_IMDB
import argparse
from dataloader.IMDB import IMDBDataset 
import utils
import h5py
# import warnings
# warnings.filterwarnings("ignore")

def backend_setting(option):
    log_dir = os.path.join(option.save_dir, option.exp_name, option.IMDB_train_mode, option.IMDB_test_mode)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if option.random_seed is None:
        option.random_seed = random.randint(1,10000)
    torch.manual_seed(option.random_seed)

    if torch.cuda.is_available() and not option.cuda:
        print('WARNING: GPU is available, but not use it')

    if not torch.cuda.is_available() and option.cuda:
        option.cuda = False

    if option.cuda:
        # torch.cuda.set_device(option.gpu)

        torch.cuda.manual_seed_all(option.random_seed)
        cudnn.benchmark = option.cudnn_benchmark
    if option.train_baseline:
        option.is_train = True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_name', default='CelebA', help='experiment name')
    # parser.add_argument('--color_var', default=0.02, type=float, help='variance for color distribution')
    # parser.add_argument('--checkpoint', default='baseline/pretraincheckpoint_step_0000.pth', help='checkpoint to resume')
    parser.add_argument('--checkpoint', default=None, help='checkpoint to resume')
    parser.add_argument('--lr', default=0.00005, type=float, help='initial learning rate')
    parser.add_argument('--random_seed', default=1, type=int, help='random seed')
    parser.add_argument('--lr_decay_period', default=3, type=int, help='lr decay period')
    parser.add_argument('--max_step', default=5, type=int, help='maximum step for training')

    parser.add_argument('--n_class', default=1, type=int, help='number of classes')
    parser.add_argument('--n_class_bias', default=1, type=int, help='number of bias classes')
    parser.add_argument('--input_size', default=224, type=int, help='input size')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--momentum', default=0.9, type=float, help='sgd momentum')
    parser.add_argument('--lr_decay_rate', default=0.1, type=float, help='lr decay rate')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='sgd optimizer weight decay')
    parser.add_argument('--seed', default=1, type=int, help='seed index')

    parser.add_argument('--log_step', default=150, type=int, help='step for logging in iteration')
    parser.add_argument('--save_step', default=1, type=int, help='step for saving in epoch')
    parser.add_argument('--save_dir', default='./results', help='save directory for checkpoint')
    parser.add_argument('--data_split', default='train', help='data split to use')
    parser.add_argument('--use_pretrain', default=False, type=bool,
                        help='whether it use pre-trained parameters if exists')
    parser.add_argument('--train_baseline', action='store_true', help='whether it train baseline or unlearning')

    parser.add_argument('--num_workers', default=4, type=int, help='number of workers in data loader')
    parser.add_argument('--cudnn_benchmark', default=True, type=bool, help='cuDNN benchmark')

    parser.add_argument('--cuda', default=True, help='enables cuda')
    parser.add_argument('-d', '--debug', action='store_true', help='debug mode')
    parser.add_argument('--is_train', default=1, type=int, help='whether it is training')
    # parser.add_argument('--gpu', default=0, type=int, help='gpu id')

    parser.add_argument('--alpha', default=1, type=int, help='alpha')
    parser.add_argument('--tau', default=10, type=int, help='tau')
    parser.add_argument('--lambda_', default=1, type=int, help='lambda')

    ## CelebA
    # parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--attributes", type=utils.str_list, default=['Blond_Hair'])
    # parser.add_argument("--predictor", type=str, default='ResNet18')
    # parser.add_argument("--beta", type=float, default=0.00025)
    # parser.add_argument("--filter", type=str, default='attGAN')
    parser.add_argument("--eval_mode", type=str, default='unbiased')

    ## IMDB
    parser.add_argument("--IMDB_train_mode", type=str, default='eb1')
    parser.add_argument("--IMDB_test_mode", type=str, default='unbiased')

    torch.set_num_threads(1)
    option = parser.parse_args()
    # extreme bias n_class_bias
    if option.IMDB_train_mode.endswith('ex'):
        option.n_class_bias = 1
    else:
        option.n_class_bias = 12
     
    print(option)
    backend_setting(option)
    trainer = Trainer(option)

    # get loader 
    data_folder = {
            'origin_image_feature_path': '/nas/vista-ssd01/users/jiazli/datasets/IMDB/processed_data/IMDB.h5py',
            'origin_target_dict_path': '/nas/vista-ssd01/users/jiazli/datasets/IMDB/processed_data/age_dict',
            'origin_sex_dict_path': '/nas/vista-ssd01/users/jiazli/datasets/IMDB/processed_data/sex_dict',
            'origin_eb1_key_list_path': '/nas/vista-ssd01/users/jiazli/datasets/IMDB/processed_data/eb1_img_list',
            'origin_eb2_key_list_path': '/nas/vista-ssd01/users/jiazli/datasets/IMDB/processed_data/eb2_img_list',
            'origin_unbiased_key_list_path': '/nas/vista-ssd01/users/jiazli/datasets/IMDB/processed_data/test_img_list',
            'augment': False
        }

    image_feature = h5py.File(data_folder['origin_image_feature_path'], 'r')
    target_dict = utils.load_pkl(data_folder['origin_target_dict_path'])
    sex_dict = utils.load_pkl(data_folder['origin_sex_dict_path'])
    eb1_key_list = utils.load_pkl(data_folder['origin_eb1_key_list_path'])
    eb2_key_list = utils.load_pkl(data_folder['origin_eb2_key_list_path'])
    unbiased_key_list = utils.load_pkl(data_folder['origin_unbiased_key_list_path'])

    import torchvision.transforms as transforms
    transform_train = transforms.Compose([
                                        transforms.Resize((224,224)),
                                        transforms.CenterCrop(180),
                                        transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                        # normalize,
                        ])
    train_set = IMDBDataset(image_feature, target_dict, sex_dict, option.IMDB_train_mode, eb1_key_list, eb2_key_list, unbiased_key_list, 'train', transform_train)
    dev_set = IMDBDataset(image_feature, target_dict, sex_dict, option.IMDB_test_mode, eb1_key_list, eb2_key_list, unbiased_key_list, 'dev_test', transform_train)

    trainval_loader = torch.utils.data.DataLoader(train_set, batch_size=option.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=option.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    if option.is_train == 1:
        save_option_IMDB(option)
        trainer.train(trainval_loader, dev_loader)
        # trainer.train(trainval_loader, testloader)
    else:
        trainer._validate(trainval_loader)
        pass

if __name__ == '__main__': main()