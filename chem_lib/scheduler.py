# coding:utf-8
from torch_geometric.nn import GAE
import torch
import torch_geometric.transforms as T
import torch_geometric.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from .localgcnconv import GCNEncoder, InnerProductWDecoder, InnerProductDecoder, MLPDecoder, GCNEnc
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy import sparse
import math


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def const(tensor, value):
    if tensor is not None:
        tensor.data.fill_(value)


class Scheduler(torch.nn.Module):
    def __init__(self, N, buffer_size, grad_indexes, use_deepsets=True, pi='naive', device='cpu'):
        super(Scheduler, self).__init__()

        self.buffer_size = buffer_size
        self.grad_indexes = grad_indexes
        self.use_deepsets = use_deepsets
        self.tasks = []

        out_channels = 32
        num_features = 256

        self.stask = N
        self.pi = pi
        self.device = device

        if self.pi == 'graph':
            self.task_vec = torch.nn.Parameter(torch.Tensor(self.stask, 1))
            glorot(self.task_vec)
            self.gae_encoder = GCNEncoder(num_features, out_channels).to(self.device)

            self.w = torch.nn.Parameter(torch.Tensor(32, 1))
            self.b = torch.nn.Parameter(torch.Tensor(1))
            glorot(self.w)
            zeros(self.b)

            self.map = torch.nn.Parameter(torch.Tensor(256, 32))
            self.mapb = torch.nn.Parameter(torch.Tensor(32))
            glorot(self.map)
            zeros(self.mapb)

    def stack_representation(self, source_proto, target_proto):
        targetstack = torch.cat(list(target_proto.values()), dim=0)
        sourcestack = torch.cat(list(source_proto.values()), dim=0)
        s_num = sourcestack.shape[0]
        t_num = targetstack.shape[0]
        return torch.cat((sourcestack, targetstack), dim=0).detach(), s_num, t_num

    def init_adj(self, representations, row, col, numpy=False):
        estimated_adj = torch.zeros(row+col, row+col)
        summi = 0
        summpercent = []
        for i in range(row):
            for j in range(col):
                xi = representations[i]
                xj = representations[row + j]
                dotinner = torch.mul(xi, xj.reshape(1, -1))
                outputij = torch.sum(dotinner,  dim=1)
                estimated_weights = F.relu(outputij)
                estimated_adj[i, row+j] = estimated_weights  # 0, 12
                estimated_adj[row + j, i] = estimated_weights  # 12, 0
                summi += estimated_weights
                summpercent.append(estimated_weights)
        torch.set_printoptions(threshold=10_000)
        md = torch.quantile(torch.tensor(summpercent), 0.9)
        adj = (estimated_adj > md).float().detach().numpy()
        if numpy:
            return adj
        else:
            adj = sparse.csr_matrix(adj)
            edge_index, edge_weight = from_scipy_sparse_matrix(adj)
            return edge_index.to(self.device)

    def sample_task(self, prob, size, replace=True):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.m = Categorical(prob)  # continue the gradient from prob
        if device == 'cuda':
            p = prob.detach().cpu().numpy()
        else:
            p = prob.detach().numpy()
        if len(np.where(p > 0)[0]) < size:
            print("exceptionally all actions")
            actions = torch.tensor(np.where(p > 0)[0])
        else:
            actions = np.random.choice(np.arange(len(prob)), p=p / np.sum(p), size=size,
                                       replace=replace)
            if device == 'cuda':
                actions = [torch.tensor(x).cuda() for x in actions]
            else:
                actions = [torch.tensor(x).cpu() for x in actions]
        return actions

    def get_weight(self, source_proto, target_proto):
        if self.pi == 'graph':
            return self.get_weight_graph(source_proto, target_proto)

    def get_weight_graph(self, source_proto, target_proto):
        x_representation, s_num, t_num = self.stack_representation(source_proto, target_proto)
        A = self.init_adj(x_representation, s_num, t_num)
        emb = self.gae_encoder(x_representation, A)

        target = torch.mean(emb[self.stask:], dim=0)
        target_rep = target.unsqueeze(0).repeat(self.stask, 1).detach()
        source = emb[:self.stask]
        source_target = source.mul(target_rep)

        soutar = torch.matmul(source_target, self.w) + self.b
        row_add = 0.5 * F.sigmoid(soutar) + 0.5 * F.sigmoid(self.task_vec)

        return None, None, row_add.view(-1, 1)

    def split4sample(self, input, slice_index):
        sliced = input[0:slice_index, slice_index: ]
        return sliced
