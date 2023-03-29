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
        self.q_encoder.fc = get_projector(self.encoder_q.fc.in_features, proj_dim, proj_dim)
        self.t_encoder.fc = get_projector(self.encoder_q.fc.in_features, proj_dim, proj_dim)

        # prediction head
        self.prediction_head = get_projector(proj_dim, proj_dim, hidden_dim)


