import argparse
import torch
from torchvision import datasets, transforms
from data_list import ImageList
#sys.path.append('../utils/')
#from utils.utils import dense_to_one_hot
import os
import numpy as np

def dense_to_one_hot_me(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros(32)
    labels_one_hot[labels_dense] = 1
    return labels_one_hot

def load_webcam():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_source='/media/zrway/8T/HJK/datasets/office31/webcam/images'
    source_list = '/media/zrway/8T/HJK/datasets/office31/list_webcam.txt'

    train_loader = torch.utils.data.DataLoader(
        	ImageList(open(source_list).readlines(), transform=transforms.Compose([
                           transforms.Resize((28,28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                      	 ]), mode='L'),
        	batch_size=128, shuffle=True, num_workers=1, drop_last=True)
    len_source = len(train_loader)
    
    for batch_idx in range(len_source):
        if batch_idx % len_source == 0:
            iter_source = iter(train_loader)    
        data_source, label_source = iter_source.next()
        #data_source, label_source = data_source.cuda(), label_source.cuda()
        label_source_vec = dense_to_one_hot_me(label_source)
    return data_source, label_source_vec, data_source, label_source_vec


