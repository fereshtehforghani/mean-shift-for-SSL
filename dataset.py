import torch
import torch.nn as nn
import random
from PIL import ImageFilter
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class ApplyTransform:
    def __init__(self, t_t, q_t):
        self.q_t = q_t
        self.t_t = t_t
        print('======= Query transform =======')
        print(self.q_t)
        print('===============================')
        print('======= Target transform ======')
        print(self.t_t)
        print('===============================')

    def __call__(self, x):
        q = self.q_t(x)
        t = self.t_t(x)
        return [q, t]

class ImageFolderEx(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super(ImageFolderEx, self).__getitem__(index)
        return index, sample, target
    
def data_loader(settings):
    ## ToDO: we have to change this part to load our own dataset
    traindir = os.path.join(settings.data, 'train')
    image_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    strong_aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    weak_aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    if settings.augmentation == 'weak/strong':
        train_dataset = ImageFolderEx(
            traindir,
            ApplyTransform(t_t=weak_aug, q_t=strong_aug)
        )
    elif settings.augmentation == 'weak/weak':
        train_dataset = ImageFolderEx(
            traindir,
            ApplyTransform(t_t=weak_aug, q_t=weak_aug)
        )
    elif settings.augmentation == 'strong/weak':
        train_dataset = ImageFolderEx(
            traindir,
            ApplyTransform(t_t=strong_aug, q_t=weak_aug)
        )
    else: # strong/strong
        train_dataset = ImageFolderEx(
            traindir,
            ApplyTransform(t_t=strong_aug, q_t=strong_aug)
        )
    
    # if settings.dataset == 'imagenet100':
    #     subset_classes(train_dataset, num_classes=100)

    print('==> train dataset')
    print(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=settings.num_workers, pin_memory=True, drop_last=True)

    return train_loader