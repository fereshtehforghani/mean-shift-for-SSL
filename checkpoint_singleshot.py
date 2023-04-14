import os
import random
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from utils.util import *
from  matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def get_feats(loader, model):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    feats, labels, ptr = None, None, 0

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(loader):
            if i == 1000:
                break
            images = images.cuda(non_blocking=True)
            cur_targets = target.cpu()
            cur_feats = normalize(model(images)).cpu()
            B, D = cur_feats.shape
            inds = torch.arange(B) + ptr

            if not ptr:
                feats = torch.zeros((len(loader.dataset), D)).float()
                labels = torch.zeros(len(loader.dataset)).long()

            feats.index_copy_(0, inds, cur_feats)
            labels.index_copy_(0, inds, cur_targets)
            ptr += B

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print(progress.display(i))

    return feats, labels

def normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)

def get_knn_dist(feats, batch_size, k):
    average = 0
    dist = torch.cdist(feats, feats, p=2)
    dist, nn_index = torch.topk(dist, k, dim=1, largest=False)
    mean_dist = torch.mean(dist, dim=1)
    average += torch.sum(mean_dist)
    return average / len(feats)
    # for i in range (0, len(feats), batch_size):
    #     dist = torch.cdist(feats[i: i+batch_size], feats[i : i+batch_size], p=2)
    #     dist, nn_index = torch.topk(dist, k, dim=1, largest=False)
    #     mean_dist = torch.mean(dist, dim=1)
    #     average += torch.sum(mean_dist)  
    # return average / (len(feats) / batch_size)

def main():
    traindir = os.path.join('my_datasets/imagenet', 'train')
    valdir = os.path.join('my_datasets/imagenet', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    batch_size=100
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, val_transform),
        batch_size=batch_size, shuffle=True,
        num_workers=16, pin_memory=True)

    model = models.__dict__['resnet50']()
    model.fc = nn.Sequential()

    wts_paths = ['output/checkpoints/cvil2/ckpt_epoch_10.pth', 'output/checkpoints/cvil2/ckpt_epoch_50.pth', 
                 'output/checkpoints/cvil2/ckpt_epoch_100.pth', 'output/checkpoints/cvil2/ckpt_epoch_150.pth',
                 'output/checkpoints/cvil2/ckpt_epoch_200.pth']    
    # wts_path = 'output/checkpoints/cvil2/ckpt_epoch_200.pth'
    means = []
    for wts_path in wts_paths:
        wts = torch.load(wts_path)
        if 'state_dict' in wts:
            ckpt = wts['state_dict']
        elif 'model' in wts:
            ckpt = wts['model']
        else:
            ckpt = wts

        for p in model.parameters():
            p.requires_grad = False

        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        ckpt = {k.replace('q_encoder.', ''): v for k, v in ckpt.items()}
        ckpt = {k.replace('t_encoder.', ''): v for k, v in ckpt.items()}
        state_dict = {}

        for m_key, m_val in model.state_dict().items():
            if m_key in ckpt:
                state_dict[m_key] = ckpt[m_key]
            else:
                state_dict[m_key] = m_val
                print('not copied => ' + m_key)

        model.load_state_dict(state_dict)
        # print(model)

        backbone = nn.DataParallel(model).cuda()
        backbone.eval()

        train_feats, _ = get_feats(train_loader, backbone)
        train_feats.to('cuda')
        average = get_knn_dist(train_feats, batch_size, k=10)
        means.append(average.item())
        print(average)
    plt.plot([10, 50, 100, 150, 200], means, label='mean', color='red', marker='o')
    plt.savefig("plot_checkpoint_.png")

if __name__ == '__main__':
    main()