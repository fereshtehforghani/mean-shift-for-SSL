import argparse

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

