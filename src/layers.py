import torch
import torch.nn as nn
from torch import FloatTensor
from torch.nn.parameter import Parameter
from scipy.spatial.distance import pdist, squareform
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TemporalModelling(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout: float, attn_mask: torch.Tensor = None, ):
        super(TemporalModelling, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, dropout, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))

class DistanceAdj(nn.Module):
    '''
    G = exp(−|γ(i − j)^2 + β|)
    '''
    def __init__(self, sigma, bias,device="cuda:1"):
        super(DistanceAdj, self).__init__()
        # self.sigma = sigma
        # self.bias = bias
        self.w = nn.Parameter(torch.FloatTensor(1))
        self.b = nn.Parameter(torch.FloatTensor(1))
        self.w.data.fill_(sigma)
        self.b.data.fill_(bias)
        self.device = device
    def forward(self, batch_size, seq_len):
        arith = np.arange(seq_len).reshape(-1, 1)
        dist = pdist(arith, metric='cityblock').astype(np.float32)
        dist = torch.from_numpy(squareform(dist)).to(self.device)
        #dist = torch.from_numpy(squareform(dist))
        # dist = torch.exp(-self.sigma * dist ** 2)
        dist = torch.exp(-torch.abs(self.w * dist ** 2 - self.b))
        dist = torch.unsqueeze(dist, 0).repeat(batch_size, 1, 1)

        return dist

class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            output = x * (1 + self.weight)
        else:
            output = x * self.weight
        return output
class GraphConvolution_att(nn.Module):
    """
    GAT: mix version of resdiual+attention of GAT; fast batch
    """

    def __init__(self, in_features, out_features, dropout=0.6, bias=False, alpha=0.2, residual=True, concat=True):
        super(GraphConvolution_att, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight = Parameter(torch.empty(in_features, out_features), requires_grad=True)

        self.concat = concat
        self.alpha = alpha
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if bias:
            self.bias = Parameter(FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        if not residual:
            self.residual = lambda x: 0
        elif (in_features == out_features):
            self.residual = lambda x: x
        else:
            # self.residual = linear(in_features, out_features)
            self.residual = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=5, padding=2)

    def reset_parameters(self):
        # stdv = 1. / sqrt(self.weight.size(1))
        nn.init.xavier_uniform_(self.weight, gain=np.sqrt(2.0))
        nn.init.xavier_uniform_(self.a, gain=np.sqrt(2.0))

        if self.bias is not None:
            self.bias.data.fill_(0.1)

    def forward(self, input, adj):
        # To support batch operations
        Wh = input.matmul(self.weight)  # [200,128]*[128,32]=[200,32]
        e = self.prepare_batch(Wh)  # (batch_zize, number_nodes, number_nodes)

        # (batch_zize, number_nodes, number_nodes)
        zero_vec = -9e15 * torch.ones_like(e)

        # (batch_zize, number_nodes, number_nodes)
        attention = torch.where(adj > 0, e, zero_vec)

        # (batch_zize, number_nodes, number_nodes)
        attention = F.softmax(attention, dim=-1)

        # (batch_zize, number_nodes, number_nodes)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # batched matrix multiplication (batch_zize, number_nodes, out_features)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            output = F.elu(h_prime)
        else:
            output = h_prime

        # output = adj.matmul(support) #[200,200]*[200,32] = [200,32]

        if self.bias is not None:
            output = output + self.bias
        if self.in_features != self.out_features and self.residual:
            input = input.permute(0, 2, 1)
            res = self.residual(input)
            res = res.permute(0, 2, 1)
            output = output + res
        else:
            output = output + self.residual(input)

        return output

    def prepare_batch(self, Wh):
        """
        with batch training
        :param Wh: (batch_zize, number_nodes, out_features)
        :return:
        """
        # Wh.shape (B, N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (B, N, 1)
        # e.shape (B, N, N)

        B, N, E = Wh.shape  # (B, N, N)

        # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)

        # broadcast add (B, N, 1) + (B, 1, N)
        e = Wh1 + Wh2.permute(0, 2, 1)  # (B, N, N)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GATMultiHeadLayer(nn.Module):
    def __init__(self,in_dim,out_dim,dropout=0.6,alpha=0.2,nheads=1,concat = True):
        super(GATMultiHeadLayer, self).__init__()
        self.dropout = dropout
        self.concat = concat
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = nheads
        self.attentions = [GraphConvolution_att(in_dim,out_dim,dropout=dropout,alpha=alpha,residual=True,concat=True) for _ in range(nheads)]
        for i,attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i),attention)

    def forward(self,x,adj):
        if self.concat:
            out = torch.cat([att(x,adj) for  att in self.attentions],dim=-1)
            
        else:
            out = torch.cat([att(x,adj).unsqueeze(0) for  att in self.attentions],dim=0)
            out = torch.mean(out,dim=0)
          
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_dim}, '
                f'{self.out_dim*self.heads}, heads={self.heads})')