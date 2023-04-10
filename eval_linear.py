import argparse
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
from arg_parser import *
from utils.util import *
from torchvision.models.alexnet import AlexNet
from torchvision.models.mobilenet import MobileNetV2


best_acc1 = 0


def main():
    global logger

    args = parse_eval_linear_args()
    create_path(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args)

def load_weights(model, wts_path):
    wts = torch.load(wts_path)
    # pdb.set_trace()
    if 'state_dict' in wts:
        ckpt = wts['state_dict']
    elif 'model' in wts:
        ckpt = wts['model']
    else:
        ckpt = wts



    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    state_dict = {}

    for m_key, m_val in model.state_dict().items():
        if m_key in ckpt:
            state_dict[m_key] = ckpt[m_key]
        else:
            state_dict[m_key] = m_val
            print('not copied => ' + m_key)

    model.load_state_dict(state_dict)
    print(model)


def get_model(arch, wts_path):
    if arch == 'alexnet':
        model = AlexNet()
        model.fc = nn.Sequential()
        load_weights(model, wts_path)
    elif arch == 'pt_alexnet':
        model = models.alexnet()
        classif = list(model.classifier.children())[:5]
        model.classifier = nn.Sequential(*classif)
        load_weights(model, wts_path)
    elif arch == 'mobilenet':
        model = MobileNetV2()
        model.fc = nn.Sequential()
        load_weights(model, wts_path)
    elif 'resnet' in arch:
        model = models.__dict__[arch]()
        model.fc = nn.Sequential()
        load_weights(model, wts_path)
    else:
        raise ValueError('arch not found: ' + arch)

    for p in model.parameters():
        p.requires_grad = False

    return model


def main_worker(args):
    global best_acc1
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
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


    train_dataset = datasets.ImageFolder(traindir, train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    train_val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, val_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    backbone = get_model(args.arch, args.weights)
    backbone = nn.DataParallel(backbone).cuda()
    backbone.eval()


    cached_feats = '%s/var_mean.pth.tar' % args.save
    if not os.path.exists(cached_feats):
        train_feats, _ = get_feats(train_val_loader, backbone, args)
        train_var, train_mean = torch.var_mean(train_feats, dim=0)
        torch.save((train_var, train_mean), cached_feats)
    else:
        train_var, train_mean = torch.load(cached_feats)

    linear = nn.Sequential(
        Normalize(),
        FullBatchNorm(train_var, train_mean),
        nn.Linear(get_channels(args.arch), len(train_dataset.classes)),
    )
    linear = linear.cuda()

    optimizer = torch.optim.SGD(linear.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    sched = [int(x) for x in args.lr_schedule.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=sched
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            linear.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, backbone, linear, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, backbone, linear, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, backbone, linear, args)

        # modify lr
        lr_scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        state = {
            'epoch': epoch + 1,
            'state_dict': linear.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }
        args.checkpoint_path = "output/mean_shift_default"
        if is_best:
            save_file = os.path.join(args.checkpoint_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            del state
            torch.cuda.empty_cache()


class Normalize(nn.Module):
    def forward(self, x):
        return x / x.norm(2, dim=1, keepdim=True)


class FullBatchNorm(nn.Module):
    def __init__(self, var, mean):
        super(FullBatchNorm, self).__init__()
        self.register_buffer('inv_std', (1.0 / torch.sqrt(var + 1e-5)))
        self.register_buffer('mean', mean)

    def forward(self, x):
        return (x - self.mean) * self.inv_std


def get_channels(arch):
    if arch == 'alexnet':
        c = 4096
    elif arch == 'pt_alexnet':
        c = 4096
    elif arch == 'resnet50':
        c = 2048
    elif arch == 'resnet18':
        c = 512
    elif arch == 'mobilenet':
        c = 1280
    else:
        raise ValueError('arch not found: ' + arch)
    return c


def train(train_loader, backbone, linear, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    backbone.eval()
    linear.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = backbone(images)
        output = linear(output)
        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(progress.display(i))


def validate(val_loader, backbone, linear, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    backbone.eval()
    linear.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = backbone(images)
            output = linear(output)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(progress.display(i))

        # TODO: this should also be done with the ProgressMeter
        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


def get_feats(loader, model, args):
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

            if i % args.print_freq == 0:
                logger.info(progress.display(i))

    return feats, labels


if __name__ == '__main__':
    main()