import argparse
from utils.util import *
def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for training')
    parser.add_argument('data', type=str, help='Path to dataset')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'imagenet100'],
                        help='Use full or subset of the dataset (default: imagenet)')
    #parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    parser.add_argument('--print_freq', type=int, help='Print frequency (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--num_workers', type=int, default=24, help='Number of workers to use (default: 24)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs (default: 200)')

    # Optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate (default: 0.01)')
    # parser.add_argument('--lr_decay_epochs', type=str, default='90,120',
    #                     help='Where to decay learning rate, can be a list (default: 90,120)')
    # parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='Decay rate for learning rate (default: 0.2)')
    #parser.add_argument('--cos', action='store_true', help='Enable cosine learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (default: 1e-4)')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')

    # Model definition
    parser.add_argument('--arch', type=str, default='alexnet',
                        choices=['alexnet', 'resnet18', 'resnet50', 'mobilenet'],
                        help='Model architecture (default: alexnet)')

    # Mean Shift
    parser.add_argument('--momentum', type=float, default=0.99, help='Momentum (default: 0.99)')
    parser.add_argument('--memory_bank_size', type=int, default=128000, help='Memory bank size (default: 128000)')
    parser.add_argument('--k', type=int, default=5, help='Top-k (default: 5)')
    parser.add_argument('--augmentation', type=str, default='weak/strong',
                        choices=['weak/strong', 'weak/weak', 'strong/weak', 'strong/strong'],
                        help='Augmentation type (default: weak/strong)')

    parser.add_argument('--weights', type=str, help='Weights to initialize the model from')
    parser.add_argument('--resume', default='', type=str,
                        help='Path to latest checkpoint (default: none)')
    parser.add_argument('--restart', action='store_true', help='Restart training from scratch')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')

    parser.add_argument('--checkpoint_path', default='output/mean_shift_default', type=str,
                        help='Path to save checkpoints (default: output/mean_shift_default)')

    opt = parser.parse_args()

    # iterations = opt.lr_decay_epochs.split(',')
    # opt.lr_decay_epochs = [int(it) for it in iterations]

    return opt

def parse_eval_linear_args():
    parser = argparse.ArgumentParser(description='Unsupervised distillation')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-a', '--arch', default='resnet50',
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=90, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--save', default='./output/distill_1', type=str,
                        help='experiment output directory')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--weights', dest='weights', type=str, required=True,
                        help='pre-trained model weights')
    parser.add_argument('--lr_schedule', type=str, default='15,30,40',
                        help='lr drop schedule')

    opt = parser.parse_args()
    
    return opt

def parse_eval_knn_args():
    parser = argparse.ArgumentParser(description='NN evaluation')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'imagenet100', 'imagenet-lt'],
                        help='use full or subset of the dataset')
    parser.add_argument('-j', '--workers', default=8, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                            choices=['alexnet' , 'resnet18' , 'resnet50', 'mobilenet' ,
                                    'l_resnet18', 'l_resnet50', 
                                    'two_resnet50', 'one_resnet50', 
                                    'moco_alexnet' , 'moco_resnet18' , 'moco_resnet50', 'moco_mobilenet', 'resnet50w5', 'teacher_resnet18',  'teacher_resnet50',
                                    'sup_alexnet' , 'sup_resnet18' , 'sup_resnet50', 'sup_mobilenet', 'pt_alexnet', 'swav_resnet50', 'byol_resnet50'])
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-p', '--print-freq', default=90, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--save', default='./output/cluster_alignment_1', type=str,
                        help='experiment output directory')
    parser.add_argument('--weights', dest='weights', type=str,
                        help='pre-trained model weights')
    parser.add_argument('--load_cache', action='store_true',
                        help='should the features be recomputed or loaded from the cache')
    parser.add_argument('-k', default=1, type=int, help='k in kNN')
    parser.add_argument('--debug', action='store_true', help='whether in debug mode or not')

    opt = parser.parse_args()
    
    return opt
