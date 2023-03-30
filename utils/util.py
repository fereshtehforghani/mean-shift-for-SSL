import os
import shutil
import torch

"""Computes and stores the average and current value"""
class AverageMeter(object):
    def __init__(self, name):
        self.name = name
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
        fmtstr = '{name} {val' + self.val + '} ({avg' + self.avg + '})'
        return fmtstr.format(**self.__dict__)

"""Computes and stores the average and current value"""
class AverageMeter(object):
    def __init__(self, name):
        self.name = name
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
        fmtstr = '{name} {val' + self.val + '} ({avg' + self.avg + '})'
        return fmtstr.format(**self.__dict__)
    
import logging


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

# checkpoint_saver = CheckpointSaver(save_dir)
# checkpoint_saver.save_each_checkpoint(state, epoch)
# checkpoint_saver.save_checkpoint(state, is_best)




