# -*- coding: utf-8 -*-
# @Time    : 2025/5/16 16:32
# @Author  : liuyuling
# @FileName: utils.py
# @Software: PyCharm
# @E-mail  ：tianlei@stu2022.jnu.edu.cn


import os
import random

import numpy as np
import scipy.io as scio
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csc_matrix, lil_matrix
from sklearn.decomposition import PCA
from tqdm import tqdm

class dotdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def seed_everything(seed=616):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(dataset_str):
    data_mat = scio.loadmat("../{}.mat".format(dataset_str))
    h = data_mat['h']
    X = data_mat['X']
    labels = data_mat['labels']

    idx_train_list = data_mat['idx_train_list']
    idx_val_list = data_mat['idx_val_list']
    idx_test_list = data_mat['idx_val_list']

    X = normalize_features(X)

    return h, X, labels, idx_train_list, idx_val_list, idx_test_list


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    if np.where(rowsum == 0)[0].shape[0] != 0:
        indices = np.where(rowsum == 0)[0]
        for i in indices:
            rowsum[i] = float('inf')
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def normalize_sparse_hypergraph_symmetric(H):
    rowsum = np.array(H.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    D = sp.diags(r_inv_sqrt)

    colsum = np.array(H.sum(0))
    r_inv_sqrt = np.power(colsum, -1).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    B = sp.diags(r_inv_sqrt)

    Omega = sp.eye(B.shape[0])

    hx = D.dot(H).dot(Omega).dot(B).dot(H.transpose()).dot(D)

    return hx

def f(x, tau):
    return torch.exp(x / tau)
