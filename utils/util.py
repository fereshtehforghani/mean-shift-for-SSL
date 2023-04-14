import os
import shutil
import torch
import logging
import math

"""Computes and stores the average and current value"""
class AverageMeter(object):
    def __init__(self, name, floating_point=':f'):
        self.name = name
        self.floating_point = floating_point
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.floating_point + '} ({avg' + self.floating_point + '})'
        return fmtstr.format(**self.__dict__)

def adjust_learning_rate(epoch, args, optimizer):
    # Epoch_start = 1, we have to subtract 1
    epoch = epoch - 1
    new_lr = args.learning_rate * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    print('New LR: {}'.format(new_lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    
    logger = logging.getLogger()
    logging_level = logging.DEBUG if debug else logging.INFO
    
    logger.setLevel(logging_level)

    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(logging_level)
        logger.addHandler(info_file_handler)

    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging_level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        file_content = f.read()
        logger.info(f"{filepath}: {file_content}")

    for f in package_files:
        with open(f, "r") as package_f:
            package_file_content = package_f.read()
            logger.info(f"{f}: {package_file_content}")

    return logger

class CheckpointSaver:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.makedirs(save_dir)
    
    def makedirs(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def save_each_checkpoint(self, state, epoch):
        ckpt_path = os.path.join(self.save_dir, 'ckpt_%d.pth.tar' % epoch)
        torch.save(state, ckpt_path)

    def save_checkpoint(self, state, is_best):
        ckpt_path = os.path.join(self.save_dir, 'checkpoint.pth.tar')
        torch.save(state, ckpt_path)
        if is_best:
            best_ckpt_path = os.path.join(self.save_dir, 'model_best.pth.tar')
            shutil.copyfile(ckpt_path, best_ckpt_path)

model_names = ['resnet50']

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [f"{self.prefix}{self.batch_fmtstr.format(batch)}"]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = f"{{:0{num_digits}d}}"
        return f"[{fmt}/{fmt.format(num_batches)}]"

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = [correct[:k].contiguous().view(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size) for k in topk]

        return res

"""Computes and stores the average and current value"""
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


