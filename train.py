import os
from arg_parser import *
from dataset import *
from MSF import *
import torch.backends.cudnn as cudnn
from utils.util import *
import time

def train_epoch(epoch, train_loader, MSF_model, optimizer, args):
    """
    Training for one epoch
    """
    MeanShift.train()
    
    batch_train_time = AverageMeter(name='batch_train_time')
    loss_meter = AverageMeter(name='loss')
    purity_meter = AverageMeter(name='purity')

    start_time = time.time()
    for idx, (indices, (q_img, t_img), labels) in enumerate(train_loader):
        # put data to gpu
        q_img = q_img.cuda(non_blocking=True)
        t_img = t_img.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # training process
        loss, purity = MSF_model(im_q=q_img, im_t=t_img, labels=labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update meters
        loss_meter.update(loss.item(), q_img.size(0))
        purity_meter.update(purity.item(), q_img.size(0))

        torch.cuda.synchronize()
        batch_train_time.update(time.time() - start_time)
        start_time = time.time()

        # print info
        if (idx + 1) % 100 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'purity {purity.val:.3f} ({purity.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_train_time,
                   purity=purity_meter,
                   loss=loss_meter))
            sys.stdout.flush()
            sys.stdout.flush()

    return loss_meter.avg

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