import os
from arg_parser import *
from dataset import *
from MSF import *
import torch.backends.cudnn as cudnn

def train_epoch(epoch, train_loader, MSF_model, optimizer, args):
    """
    Training for one epoch
    """
    MeanShift.train()

def main():
    args = parse_args()
    args.start_epoch = 1

    print(args)
    os.makedirs(args.checkpoint_path, exist_ok=True)

    train_dataloader = data_loader(args)

    #create MSF model
    MSF_model = MeanShift(k=args.k, m=args.momentum, memory_bank_size=args.memory_bank_size)

    # Data parallel
    MSF_model.q_encoder = torch.nn.DataParallel(MSF_model.q_encoder)
    MSF_model.t_encoder = torch.nn.DataParallel(MSF_model.t_encoder)
    MSF_model.q_prediction_head = torch.nn.DataParallel(MSF_model.q_prediction_head)

    # Move to default device
    MSF_model = MSF_model.cuda()
    cudnn.benchmark = True

    parameters = [param for param in MSF_model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=args.learning_rate, momentum=args.sgd_momentum, weight_decay=args.weight_decay)

    # resume from a checkpoint
    if args.weights:
        print('Load from checkpoint: {}'.format(args.weights))
        ckpt = torch.load(args.weights)
        print('Resume from epoch: {}'.format(ckpt['epoch']))
        if 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt['state_dict']
        msg = MSF_model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        args.start_epoch = ckpt['epoch'] + 1
    
    if args.resume:
        print('Resume from checkpoint: {}'.format(args.resume))
        ckpt = torch.load(args.resume)
        print('Resume from epoch: {}'.format(ckpt['epoch']))
        MSF_model.load_state_dict(ckpt['state_dict'], strict=True)
        if not args.restart:
            optimizer.load_state_dict(ckpt['optimizer'])
            args.start_epoch = ckpt['epoch'] + 1
    
    # train loop



if __name__ == '__main__':
    main()