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
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, val_transform),
        batch_size=256, shuffle=True,
        num_workers=16, pin_memory=True,
    )

    model = models.__dict__['resnet50']()
    model.fc = nn.Sequential()

    # wts_path = 'output/checkpoints/cvil2/pretrained-checkpoint/msf_weak_strong_aug_topk_10_mbs_1024k_resnet50_inet1m_nn_62.5.pth'
    wts_paths = ['output/checkpoints/cvil2/ckpt_epoch_10.pth', 'output/checkpoints/cvil2/ckpt_epoch_50.pth', 
                 'output/checkpoints/cvil2/ckpt_epoch_100.pth', 'output/checkpoints/cvil2/ckpt_epoch_150.pth',
                 'output/checkpoints/cvil2/ckpt_epoch_200.pth']
    random_labels = torch.randint(0, 1000, (10,))
    for i in range(len(wts_paths)):
        wts_path = wts_paths[i]
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
        # ckpt = {k.replace('encoder_q.', ''): v for k, v in ckpt.items()}
        # ckpt = {k.replace('encoder_k.', ''): v for k, v in ckpt.items()}

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
        print(model)

        backbone = nn.DataParallel(model).cuda()
        backbone.eval()

        train_feats, train_label = get_feats(train_loader, backbone)
        indices = torch.where(torch.isin(train_label, random_labels))

        X_embedded = TSNE(n_components=2).fit_transform(train_feats[indices[0]])

        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=train_label[indices[0]], s=1)
        plt.savefig("plot_" + str(i) + ".png")
        plt.clf()

if __name__ == '__main__':
    main()