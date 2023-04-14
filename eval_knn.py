import builtins
from collections import Counter, OrderedDict
from random import shuffle
import argparse
import os
import random
import shutil
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
import numpy as np
import faiss
from utils.util import *
from torchvision.models.resnet import resnet50
from arg_parser import *

tempreture = 0.04


def main():
    global logger

    args = parse_eval_knn_args()
    create_path(args.save)

    if not args.debug:
        logger = get_logger(
            logpath=os.path.join(args.save, 'logs'),
            # logpath=os.path.join(args.save, 'knn.logs'),
            filepath=os.path.abspath(__file__)
        )
        def print_pass(*args):
            logger.info(*args)
        builtins.print = print_pass

    print(args)

    main_worker(args)


def get_model(args):

    model = resnet50()
    model.fc = nn.Sequential()
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.weights)
    if 'model' in checkpoint:
        sd = checkpoint['model']
    else:
        sd = checkpoint['state_dict']
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if 'fc' not in k}
    sd = {k: v for k, v in sd.items() if 'encoder_k' not in k}
    sd = {k.replace('encoder_q.', ''): v for k, v in sd.items()}
    sd = {('module.'+k): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(model)
    print(msg)
    for param in model.parameters():
        param.requires_grad = False

    return model


class ImageFolderEx(datasets.ImageFolder) :
    def __getitem__(self, index):
        sample, target = super(ImageFolderEx, self).__getitem__(index)
        return index, sample, target


def get_data_loaders(dataset_dir, bs, workers, dataset='imagenet'):
    traindir = os.path.join(dataset_dir, 'train')
    valdir = os.path.join(dataset_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    augmentation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = ImageFolderEx(traindir, augmentation)
    val_dataset = ImageFolderEx(valdir, augmentation)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader


def main_worker(args):
    start = time.time()
    train_loader, val_loader = get_data_loaders(args.data, args.batch_size, args.workers, args.dataset)
    model = get_model(args)
    cudnn.benchmark = True
    cached_feats = '%s/train_feats.pth.tar' % args.save
    if args.load_cache and os.path.exists(cached_feats):
        print('load train feats from cache =>')
        train_feats, train_labels, train_inds = torch.load(cached_feats)
    else:
        print('get train feats =>')
        train_feats, train_labels, train_inds = get_features(train_loader, model, args.print_freq)
        torch.save((train_feats, train_labels, train_inds), cached_feats)

    cached_feats = '%s/val_feats.pth.tar' % args.save
    if args.load_cache and os.path.exists(cached_feats):
        print('load val feats from cache =>')
        val_feats, val_labels, val_inds = torch.load(cached_feats)
    else:
        print('get val feats =>')
        val_feats, val_labels, val_inds = get_features(val_loader, model, args.print_freq)
        torch.save((val_feats, val_labels, val_inds), cached_feats)

    train_feats = l2_normalize(train_feats)
    val_feats = l2_normalize(val_feats)

    for k in [1,20]:
        print(k)
        acc = faiss_knn(train_feats, train_labels, val_feats, val_labels, k)
        nn_time = time.time() - start
        print('=> time : {:.2f}s'.format(nn_time))
        print(' * Acc {:.2f}'.format(acc))


def l2_normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


def faiss_knn(feats_train, targets_train, feats_val, targets_val, k):
    feats_train = feats_train.numpy()
    targets_train = targets_train.numpy()
    feats_val = feats_val.numpy()
    targets_val = targets_val.numpy()

    d = feats_train.shape[-1]

    index = faiss.IndexFlatL2(d)  # build the index
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(index, co)
    gpu_index.add(feats_train)

    D, I = gpu_index.search(feats_val, k)

    pred = np.zeros(I.shape[0], dtype=np.int)
    conf_mat = np.zeros((1000, 1000), dtype=np.int)
    for i in range(I.shape[0]):
        votes = list(Counter(targets_train[I[i]]).items())
        shuffle(votes)
        pred[i] = max(votes, key=lambda x: x[1])[0]
        conf_mat[targets_val[i], pred[i]] += 1

    acc = 100.0 * (pred == targets_val).mean()
    assert acc == (100.0 * (np.trace(conf_mat) / np.sum(conf_mat)))
    return acc


def get_features(loader, model, print_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix='Test: ')
    model.eval()
    features, labels, indices, ptr = None, None, None, 0

    with torch.no_grad():
        end = time.time()
        for i, (index, images, target) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            cur_targets = target.cpu()
            cur_feats = model(images).cpu()
            cur_indices = index.cpu()

            B, D = cur_feats.shape
            inds = torch.arange(B) + ptr

            if not ptr:
                features = torch.zeros((len(loader.dataset), D)).float()
                labels = torch.zeros(len(loader.dataset)).long()
                indices = torch.zeros(len(loader.dataset)).long()

            features.index_copy_(0, inds, cur_feats)
            labels.index_copy_(0, inds, cur_targets)
            indices.index_copy_(0, inds, cur_indices)
            ptr += B

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print(progress.display(i))

    return features, labels, indices

if __name__ == '__main__':
    main()