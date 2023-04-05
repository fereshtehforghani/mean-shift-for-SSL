import torch
import torch.nn as nn
from model import *

   
def get_projector(inp_dim, hidden_dim, out_dim):
    mlp = nn.Sequential(
        nn.Linear(inp_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )
    return mlp

def get_shuffle_ids(batch_size):
    shuffle_ids = torch.randperm(batch_size)
    reverse_ids = torch.argsort(shuffle_ids)
    return shuffle_ids.long().cuda(), reverse_ids.long().cuda()


class MeanShift(nn.Module):

    def __init__(self, k=5, m=0.99, memory_bank_size=128000):
        super().__init__()

        # initialize properties
        self.k = k
        self.m = m
        self.memory_bank_size = memory_bank_size

        # initialize encoders
        self.q_encoder = resnet50()
        self.t_encoder = resnet50()
        
        # embedding dimensions
        hidden_dim = self.q_encoder.fc.in_features * 2
        proj_dim = self.q_encoder.fc.in_features // 4

        # projection head
        self.q_encoder.fc = get_projector(self.q_encoder.fc.in_features, hidden_dim, proj_dim)
        self.t_encoder.fc = get_projector(self.t_encoder.fc.in_features, hidden_dim, proj_dim)

        # prediction head
        self.q_prediction_head = get_projector(proj_dim, hidden_dim, proj_dim)
        
        # copy query encoder to target encoder
        for q_param, t_param in zip(self.q_encoder.parameters(), self.t_encoder.parameters()):
            t_param.data.copy_(q_param.data)
            t_param.requires_grad = False
        
        print("memory-bank size {}".format(self.memory_bank_size))
        
        # initialize queue and normalize the embeddings in queue
        self.register_buffer('queue', torch.randn(self.memory_bank_size, proj_dim))
        self.queue = nn.functional.normalize(self.queue, dim=1)

        # initialize the labels queue
        self.register_buffer('labels', -1*torch.ones(self.memory_bank_size).long())
       
        # intialize the queue pointer
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
    
    def forward(self, q_img, t_img, labels):
        """
        Input:
            q_img: a batch of query images
            t_img: a batch of target images
            label: the label of the query image
        Output:
            loss: the loss for the query image
            purity: the purity of the target image
        """
        # compute query features
        q_features = self.q_encoder(q_img)
        query = self.q_prediction_head(q_features)
        q_proj = nn.functional.normalize(query, dim=1)
        
        # compute target features
        with torch.no_grad():
            # update the target encoder
            self._momentum_update_target_encoder()
            
            # shuffle targets
            shuffle_ids, reverse_ids = get_shuffle_ids(t_img.shape[0])
            t_img = t_img[shuffle_ids]

            # forward through the target encoder
            current_target = nn.functional.normalize(self.t_encoder(t_img), dim=1)

            # undo shuffle
            current_target = current_target[reverse_ids].detach()
            self._dequeue_and_enqueue(current_target, labels)

        
        # calculate mean shift regression loss
        targets = self.queue.clone().detach()
       
        # calculate distances between vectors
        # calculate distances between current target and targets in the queue
        # 2 - 2 * (current_target * targets) is equivalent to 
        # 2 - 2 * (dot product between current_target and targets) because ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a * b)
        t_dist = 2 - 2 * torch.einsum('bc,kc->bk', [current_target, targets])
        # calculate distances between query and targets in the queue
        # similar to dist_t calculation
        q_dist = 2 - 2 * torch.einsum('bc,kc->bk', [q_proj, targets])

        # select the k nearest neighbors with smallest distance based on current target
        _, nn_index = t_dist.topk(self.k, dim=1, largest=False)
        q_nn_dist = torch.gather(q_dist, 1, nn_index)

        labels = labels.unsqueeze(1).expand(q_nn_dist.shape[0], q_nn_dist.shape[1])
        labels_queue = self.labels.clone().detach().unsqueeze(0).expand((q_nn_dist.shape[0], self.memory_bank_size))
        labels_queue = torch.gather(labels_queue, dim=1, index=nn_index)
        matches = (labels_queue == labels).float()

        loss = (q_nn_dist.sum(dim=1) / self.k).mean()
        purity = (matches.sum(dim=1) / self.k).mean()

            
        return loss, purity
    
    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        """
        Momentum update of the target encoder
        """
        for q_param, t_param in zip(self.q_encoder.parameters(), self.t_encoder.parameters()):
            t_param.data = t_param.data * self.m + q_param.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, current_target, labels):
        """
        Update the queue and the labels queue
        """
        batch_size = current_target.shape[0]
        ptr = int(self.queue_ptr)
        assert self.memory_bank_size % batch_size == 0

        # replace the targets at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size] = current_target
        self.labels[ptr:ptr + batch_size] = labels
        
        # move pointer
        ptr = (ptr + batch_size) % self.memory_bank_size 

        self.queue_ptr[0] = ptr
        

        
   


