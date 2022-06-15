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


def load_img(e_num, path):
    img_dict = pickle.load(open(path, "rb"))
    # init unknown img vector with mean and std deviation of the known's
    imgs_np = np.array(list(img_dict.values()))
    mean = np.mean(imgs_np, axis=0)
    std = np.std(imgs_np, axis=0)
    # img_embd = np.array([np.zeros_like(img_dict[0]) for i in range(e_num)]) # no image
    # img_embd = np.array([img_dict[i] if i in img_dict else np.zeros_like(img_dict[0]) for i in range(e_num)])
    img_embd = np.array(
        [img_dict[i] if i in img_dict else np.random.normal(mean, std, mean.shape[0]) for i in range(e_num)])
    print("%.2f%% entities have images" % (100 * len(img_dict) / e_num))
    return img_embd



class img:
    def __init__(self,left_ents, right_ents, ills):
        self.train_ill = None
        self.left_ents = left_ents
        self.right_ents = right_ents
        self.ills = ills
        self.ent_num = len(left_ents)+len(right_ents)
        img_vec_path = "../data/EAKpkls/" + LANG + "_en_GA_id_img_feature_dict.pkl"
        self.img_features = load_img(self.ent_num, img_vec_path)
        self.img_features = F.normalize(torch.Tensor(self.img_features).cuda(CUDA_NUM))
        self.img_sim = None
        self.img_index =None
        self.candidate = None
        self.getunsup()
        self.initimgIndex()

    def tureLinkRate(self):
        count = 0.0
        for link in self.train_ill:
            if link in self.ills:
                count = count + 1
        print("%.2f%% in true links" % (count / len(self.train_ill) * 100))
        print("visual links length: %d" % (len(self.train_ill)))


    def getunsup(self):
        # if unsupervised? use image to obtain links
        l_img_f = self.img_features[self.left_ents]  # left images
        r_img_f = self.img_features[self.right_ents]  # right images

        img_sim = l_img_f.mm(r_img_f.t())
        self.img_sim = img_sim
        topk = UNSUP_K
        two_d_indices = get_topk_indices(img_sim, topk * 100)
        del l_img_f, r_img_f, img_sim
        visual_links = []
        used_inds = []
        count = 0
        for ind in two_d_indices:
            if self.left_ents[ind[0]] in used_inds: continue
            if self.right_ents[ind[1]] in used_inds: continue
            used_inds.append(self.left_ents[ind[0]])
            used_inds.append(self.right_ents[ind[1]])
            visual_links.append((self.left_ents[ind[0]], self.right_ents[ind[1]]))
            count += 1
            if count == topk:
                break
        self.train_ill = visual_links
        self.candidate = visual_links
        self.tureLinkRate()

    def initimgIndex(self):
        vals, indices = self.img_sim.topk(10)
        self.img_index = indices


    # def candidateill(self):




