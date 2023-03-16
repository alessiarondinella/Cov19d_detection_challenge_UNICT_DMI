import torch.nn as nn


class Unsqueeze(nn.Module):
    def __init__(self, dim_list):
        super(Unsqueeze, self).__init__()
        self.dim_list = dim_list
        
    def forward(self,io):
        for dim in self.dim_list:
            io = io.unsqueeze(dim)
        return io


class Squeeze(nn.Module):
    def __init__(self, dim_list):
        super(Squeeze, self).__init__()
        self.dim_list = dim_list
        
    def forward(self,io):
        for dim in self.dim_list:
            io = io.squeeze(dim)
        return io

class MultiheadAttentionMod(nn.Module):
    def __init__(self, in_dim, numHeadMultiHeadAttention):
        super(MultiheadAttentionMod, self).__init__()
        self.attention = nn.MultiheadAttention(in_dim, numHeadMultiHeadAttention, batch_first=False)
    
    def forward(self,i):
        o,_ = self.attention(i,i,i)
        return o