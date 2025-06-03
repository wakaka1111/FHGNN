# -*- coding: utf-8 -*-
# @Time    : 2025/5/16 16:32
# @Author  : liuyuling
# @FileName: run_fg.py
# @Software: PyCharm

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim

from models import HGCN
from utils import load_data, normalize_sparse_hypergraph_symmetric, seed_everything, accuracy

warnings.filterwarnings('ignore')
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dim_hidden', type=int, default=512, help='hidden dim')
parser.add_argument('--dropout', type=int, default=0.5, help='dropout probability')
parser.add_argument('--dataset', type=str, default='citeseer1000', help='dataset name')
parser.add_argument('--early_stopping', type=int, default='100', help='early_stopping')
parser.add_argument('--F', type=int, default='5', help='F')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.cuda.set_device(args.gpu_id)

# Load data
h, X, labels, idx_train_list, idx_val_list, idx_test_list = load_data(dataset_str=args.dataset)

nclass = labels.shape[1]

param = ["lr", "weight_decay", "dim_hidden", "F", "acc_train", "acc_test"]
save_name = "./result/result_" + args.dataset + "_tuning_parameters.csv"
pd.DataFrame(columns=param).to_csv(save_name, index=False)
print(f'save name:{save_name}')

labels = torch.LongTensor(np.where(labels)[1]).to(device)
orgin_X = X

h = h.copy()
h_nor = normalize_sparse_hypergraph_symmetric(h)
h = torch.from_numpy(h_nor.toarray()).float().to(device)
print(X.shape[1])
seed_everything(seed=args.seed)
X = torch.from_numpy(X.toarray()).float().to(device)
for trial in range(len(idx_train_list)):
    idx_train = torch.LongTensor(idx_train_list[trial].astype(np.int64)).to(device)
    idx_val = torch.LongTensor(idx_val_list[trial].astype(np.int64)).to(device)
    idx_test = torch.LongTensor(idx_test_list[trial].astype(np.int64)).to(device)

    for args.F in [3, 4, 5, 6, 7]:
        for args.dim_hidden in [512, 256, 128]:
            for args.lr in [1e-2, 5e-2, 1e-3, 5e-3]:
                for args.weight_decay in [5e-4, 5e-3, 1e-4, 1e-3, 0]:
                    seed_everything(seed=args.seed)

                    model = HGCN(X.shape[1], args.dim_hidden, nclass, args.dropout, args.F)
                    model.to(device)
                    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                                           weight_decay=args.weight_decay)
                    cost_val = []
                    for epoch in range(args.epochs):
                        t = time.time()
                        model.train()
                        output = model(X, h)
                        loss_train = F.nll_loss(output[idx_train], labels[idx_train])

                        acc_train = accuracy(output[idx_train], labels[idx_train])

                        optimizer.zero_grad()
                        loss_train.backward()
                        optimizer.step()

                        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
                        acc_val = accuracy(output[idx_val], labels[idx_val])
                        cost_val.append(loss_val.item())
                        print('Epoch: {:04d}'.format(epoch + 1),
                              'loss_train: {:.4f}'.format(loss_train.item()),
                              'acc_train: {:.4f}'.format(acc_train.item()),
                              'loss_val: {:.4f}'.format(loss_val.item()),
                              'acc_val: {:.4f}'.format(acc_val.item()),
                              'time: {:.4f}s'.format(time.time() - t))

                    with torch.no_grad():
                        model.eval()
                        output = model(X, h)
                        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
                        acc_test = accuracy(output[idx_test], labels[idx_test])
                        print("Train set results:", "acc_train: {:.4f}".format(acc_train.item()),
                              "acc_test: {:.4f}".format(acc_test.item()), )
                        out = [args.lr, args.weight_decay, args.dim_hidden, args.F, acc_train.item(), acc_test.item()]
                        pd.DataFrame([out]).to_csv(save_name, index=False, mode='a+', header=False)
