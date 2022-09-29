import torch
from torchvision import datasets, transforms
from dataloader import *
import numpy as np 
import os
import argparse
import torch
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--biased_var", type=float, default=0.02) # -1 uniformly
parser.add_argument("--test_mode", type=str, default='uniform')
opt = vars(parser.parse_args())
opt['data_folder'] = '/nas/vista-ssd01/users/jiazli/datasets/MNIST/'
opt['save_folder'] = '/nas/vista-ssd01/users/jiazli/datasets/CMNIST/generated_render'
opt['save_name'] = 'mnist_10color_jitter_var_{}.npy'.format(opt['biased_var'])

print("Biased Colored MNIST with variance = {}".format(opt["biased_var"]))
# load grey scale data to generate dataset
train_set_grey = datasets.MNIST(root=opt["data_folder"], train=True, download=False, transform=transforms.ToTensor())
test_set_grey = datasets.MNIST(root=opt["data_folder"], train=False, download=False, transform=transforms.ToTensor())

# # imbalance
# train_set_grey = utils.imbalance(train_set_grey, balance=4500, minority=opt['minority'], imbalance=opt['imbalance'])
# # balance dev and test dataset
# dev_set_grey = utils.imbalance(dev_set_grey, balance=900, minority=None, imbalance=None)
# test_set_grey = utils.imbalance(test_set_grey, balance=890, minority=None, imbalance=None)

train_set = ColoredDataset_generated(train_set_grey, var=opt["biased_var"])
if opt['test_mode'] == 'uniform':
    # no shuffle color 
    test_set = ColoredDataset_generated(test_set_grey, var=-1)
elif opt['test_mode'] == 'shuffle':
    # shuffle color
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=50000, shuffle=True, num_workers=4, pin_memory=True)
    train_iter = iter(train_loader)
    train_data, targets, colors = train_iter.next()
    colors = get_colors(train_data, targets)
    new_colors = colors[torch.randperm(10)]
    test_set = ColoredDataset_generated(test_set_grey, colors=new_colors, var=opt["biased_var"])

utils.distribution(train_set)
utils.distribution(test_set)


data = dict()

def process(dataset):
    imgs = []
    labels = []
    for i, (img, label, _) in enumerate(dataset):
        imgs.append(img.numpy())
        labels.append(label.numpy())
    imgs = np.array(imgs)
    labels = np.array(labels)
    imgs *= 255
    imgs = np.moveaxis(imgs, 1, 3)
    return imgs, labels
        
data['train_image'], data['train_label'] = process(train_set) 
data['test_image'], data['test_label'] = process(test_set) 

with open(os.path.join(opt['save_folder'] + '_' + opt['test_mode'], opt['save_name']), 'wb') as f:
    np.save(f, data)
