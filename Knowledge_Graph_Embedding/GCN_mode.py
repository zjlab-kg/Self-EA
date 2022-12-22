import math
import pickle
import random

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import scipy.sparse as sp
from Param import *
from GCN_basic_bert_unit.Basic_Bert_Unit_model import Basic_Bert_Unit_model


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    longTensor = torch.cuda.LongTensor
    floatTensor = torch.cuda.FloatTensor


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
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

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def get_adjr(ent_size, triples, norm=False):
    print('getting a sparse tensor r_adj...')
    M = {}
    for tri in triples:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 0
        M[(tri[0], tri[2])] += 1
    ind, val = [], []
    for (fir, sec) in M:
        ind.append((fir, sec))
        ind.append((sec, fir))
        val.append(M[(fir, sec)])
        val.append(M[(fir, sec)])
    for i in range(ent_size):
        ind.append((i, i))
        val.append(1)
    if norm:
        ind = np.array(ind, dtype=np.int32)
        val = np.array(val, dtype=np.float32)
        adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size, ent_size), dtype=np.float32)
        return sparse_mx_to_torch_sparse_tensor(normalize_adj(adj))
    else:
        M = torch.sparse_coo_tensor(torch.LongTensor(ind).t(), torch.FloatTensor(val), torch.Size([ent_size, ent_size]))
        return M


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)

        self.dropout = dropout

    def forward(self, x, feature):
        x = F.relu(self.gc1(x, feature))  # change to leaky relu
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, feature)
        return x


class combine_model(nn.Module):
    def __init__(self, ENT_NUM, triples):
        super(combine_model, self).__init__()
        n_units = [int(x) for x in UNITS.strip().split(",")]

        # input_dim = int(n_units[0])
        # entity_emb = nn.Embedding(ENT_NUM, input_dim)
        # nn.init.normal_(entity_emb.weight, std=1.0 / math.sqrt(ENT_NUM))
        # entity_emb.requires_grad = True
        with open(DATA_PATH + "feature/{}_basic.npy".format(LANG), 'rb') as f:
            d_feature = np.load(f)
        entity_emb = torch.Tensor(d_feature)

        self.entity_emb = entity_emb.cuda(CUDA_NUM)
        self.cross_graph_model = GCN(n_units[0], n_units[1], n_units[2], dropout=0).cuda(CUDA_NUM)
        self.input_idx = torch.LongTensor(np.arange(ENT_NUM)).cuda(CUDA_NUM)
        # 属性矩阵
        adj = get_adjr(ENT_NUM, triples, norm=True)
        self.adj = adj.cuda(CUDA_NUM)
        self.bert = Basic_Bert_Unit_model(MODEL_INPUT_DIM, MODEL_OUTPUT_DIM).cuda(CUDA_NUM)
        self.fc = nn.Linear(MODEL_OUTPUT_DIM + n_units[2], MODEL_OUTPUT_DIM)

    # def forward(self, batch_word_list, attention_mask, eids):
    #     dcb_emb = self.bert(batch_word_list, attention_mask)
    #     input_idx = torch.LongTensor(eids).cuda(CUDA_NUM)
    #     gph_emb = self.cross_graph_model(self.entity_emb(input_idx), dcb_emb)
    #     # gph_emb = self.cross_graph_model(self.entity_emb(self.input_idx), self.adj)
    #     joint_emb = torch.cat([
    #         F.normalize(dcb_emb),
    #         F.normalize(gph_emb)
    #     ], dim=1)
    #     return joint_emb

    def forward(self, batch_word_list, attention_mask, eids):
        dcb_emb = self.bert(batch_word_list, attention_mask)
        gph_emb = self.cross_graph_model(self.entity_emb, self.adj)
        joint_emb = torch.cat([
            F.normalize(dcb_emb),
            F.normalize(gph_emb[eids])
        ], dim=1)
        joint_emb = self.fc(joint_emb)
        return joint_emb


if __name__ == '__main__':
    with open(DATA_PATH + "feature/ja_basic.npy", 'rb') as f:
        d_feature = np.load(f)
    print(type(torch.Tensor(d_feature).cuda(CUDA_NUM)))
    entity_emb = nn.Embedding(988, 400)
    nn.init.normal_(entity_emb.weight, std=1.0 / math.sqrt(988))
    entity_emb.requires_grad = True
    print(type(entity_emb.cuda(4)))
