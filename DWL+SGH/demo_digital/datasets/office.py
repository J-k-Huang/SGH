import argparse
import torch
from torchvision import datasets, transforms
from data_list import ImageList
import os
import numpy as np
#sys.path.append('../utils/')
#from utils.utils import dense_to_one_hot


def dense_to_one_hot_me(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros(32)
    labels_one_hot[labels_dense] = 1
    return labels_one_hot

def load_office():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    source_list = '/media/zrway/8T/HJK/datasets/office31/list_amazon.txt'
    target_list = '/media/zrway/8T/HJK/datasets/office31/list_webcam.txt'

    train_loader = torch.utils.data.DataLoader(
        	ImageList(open(source_list).readlines(), transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                      	 ]), mode='L'),
        	batch_size=128, shuffle=True, num_workers=1, drop_last=True)
    train_loader1 = torch.utils.data.DataLoader(
        	ImageList(open(target_list).readlines(), transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                      	 ]), mode='L'),
        	batch_size=128, shuffle=True, num_workers=1, drop_last=True)
    len_source = len(train_loader)
    len_target = len(train_loader1)
    if len_source > len_target:
        num_iter = len_source
    else:
        num_iter = len_target
    
    for batch_idx in range(num_iter):
        if batch_idx % len_source == 0:
            iter_source = iter(train_loader)    
        if batch_idx % len_target == 0:
            iter_target = iter(train_loader1)
        data_source, label_source = iter_source.next()
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target, label_target = iter_target.next()
        data_target = data_target.cuda()

    return data_source, label_source, data_target, label_target


