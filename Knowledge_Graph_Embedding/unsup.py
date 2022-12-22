import torch
import numpy as np
import pickle
from Param import *
import torch.nn.functional as F


def get_topk_indices(M, K=1000):
    H, W = M.shape
    M_view = M.view(-1)
    vals, indices = M_view.topk(K)
    print("highest sim:", vals[0].item(), "lowest sim:", vals[-1].item())
    two_d_indices = torch.cat(((indices // W).unsqueeze(1), (indices % W).unsqueeze(1)), dim=1)
    return two_d_indices


class img:
    def __init__(self, left_ents, right_ents, ills):
        self.train_ill = None
        self.left_ents = left_ents
        self.right_ents = right_ents
        self.ills = ills
        self.candidate = None
        self.last_1 = None
        self.last_2 =None

    def tureLinkRate(self):
        count = 0.0
        for link in self.train_ill:
            if link in self.ills:
                count = count + 1
        print("%.2f%% in true links" % (count / len(self.train_ill) * 100))
        print("visual links length: %d" % (len(self.train_ill)))


