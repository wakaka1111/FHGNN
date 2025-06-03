# -*- coding: utf-8 -*-
# @Time    : 2025/5/16 16:32
# @Author  : liuyuling
# @FileName: models.py
# @Software: PyCharm

import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans
from utils import normalize_sparse_hypergraph_symmetric
from scipy.sparse import csc_matrix

class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """

    @staticmethod
    def forward(ctx, M1, M2):
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t(), g)

        return g1, g2


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_features, adj):
        support = SparseMM.apply(input_features, self.weight)
        output = SparseMM.apply(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

class FGCNNLayer(Module):
    def __init__(self, out_features, fussy):
        super(FGCNNLayer, self).__init__()
        self.multi_gaussian_membership = MultiGaussianMembership(out_features, fussy)

    def forward(self, x):
        m_ship = self.multi_gaussian_membership(x)
        m_ship_max = torch.max(m_ship, dim=2)[0] #(batch_size, out_features)
        return m_ship_max
class MultiGaussianMembership(Module):
    def __init__(self, out_features, fussy):
        super(MultiGaussianMembership, self).__init__()
        self.mu = Parameter(torch.randn(out_features, fussy))
        self.sigma = Parameter(torch.randn(out_features, fussy))

    def kmeans_init(self, data, num_gaussians):
        kmeans = KMeans(n_clusters=num_gaussians, init='k-means++', ).fit(data)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        self.mu.data = torch.tensor(centers.T, dtype=torch.float32)
        sigma = np.zeros((data.shape[1],num_gaussians))
        for i in range(num_gaussians):
            cluster_points =data[labels == i]
            if len(cluster_points) > 1:
                sigma[:, i] = np.std(cluster_points, axis=0) + 1e-5
            self.sigma.data = torch.tensor(sigma, dtype=torch.float32)

    def forward(self, x):
        batch_size, out_features = x.shape
        x_expanded = x.unsqueeze(2).expand(batch_size, out_features,
                                           self.mu.size(1))
        # 计算高斯隶属度
        sigma_clamped = torch.clamp(self.sigma, min=1e-5) #对 sigma 设置下界
        membership_degrees = torch.exp(-((x_expanded - self.mu) ** 2) / (2 * sigma_clamped ** 2))

        return membership_degrees


class HGCN(nn.Module):
    def __init__(self, input_dim, dim_hidden, nclass, dropout, fussy):
        super(HGCN, self).__init__()
        # convolution
        self.gcx1 = GraphConvolution(input_dim, dim_hidden)
        self.fc1 = FGCNNLayer(dim_hidden, fussy)
        self.gcx2 = GraphConvolution(dim_hidden, nclass)
        self.fc2 = FGCNNLayer(nclass, fussy)
        self.dropout = dropout

    def forward(self, x, h):

        x1_1 = self.gcx1(x, h)
        x1_2 = self.fc1(x1_1)
        x1 = torch.mul(x1_1, x1_2)
        x2_1 = self.gcx2(x1, h)
        x2_2 = self.fc2(x2_1)
        x2 = torch.mul(x2_1, x2_2)
        output = F.log_softmax(x2, dim=1)

        return output
