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
        hidden_dim = self.encoder_q.fc.in_features * 2
        proj_dim = self.encoder_q.fc.in_features // 4

        # projection head
        self.q_encoder.fc = get_projector(self.encoder_q.fc.in_features, hidden_dim, proj_dim)
        self.t_encoder.fc = get_projector(self.encoder_q.fc.in_features, hidden_dim, proj_dim)

        # prediction head
        self.q_prediction_head = get_projector(proj_dim, hidden_dim, hidden_dim)
        
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

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for q_param, t_param in zip(self.q_encoder.parameters(), self.t_encoder.parameters()):
            t_param.data = t_param.data * self.m + q_param.data * (1. - self.m)

        

        
   


