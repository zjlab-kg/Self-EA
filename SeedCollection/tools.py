import torch
import torch.nn.functional as F
from Param import *


def cosine_similarity(emb1, emb2, bs=128, cuda_num=0):
    """
    return cosine similarity matrix of embedding1(emb1) and embedding2(emb2)
    """
    array_emb1 = F.normalize(torch.FloatTensor(emb1), p=2, dim=1)
    array_emb2 = F.normalize(torch.FloatTensor(emb2), p=2, dim=1)
    res_mat = batch_mat_mm(array_emb1, array_emb2.t(), cuda_num, bs=bs)
    return res_mat


def batch_mat_mm(mat1, mat2, cuda_num, bs=128):
    # be equal to matmul, Speed up computing with GPU
    res_mat = []
    axis_0 = mat1.shape[0]
    for i in range(0, axis_0, bs):
        temp_div_mat_1 = mat1[i:min(i + bs, axis_0)].cuda(cuda_num)
        res = temp_div_mat_1.mm(mat2.cuda(cuda_num))
        res_mat.append(res.cpu())
    res_mat = torch.cat(res_mat, 0)
    return res_mat


def get_topk_indices(M, K=1000):
    H, W = M.shape
    M_view = M.view(-1)
    vals, indices = M_view.topk(K)
    print("highest sim:", vals[0].item(), "lowest sim:", vals[-1].item())
    two_d_indices = torch.cat(((indices // W).unsqueeze(1), (indices % W).unsqueeze(1)), dim=1)
    return two_d_indices


def calculate_correct_rate(links, ills):
    count = 0.0
    for link in links:
        if link in ills:
            count += 1
    print("true links : %.2f%%" % (count / len(links) * 100))
    if Unsup:
        with open(ResultFile, 'a') as f:
            f.write('UNSUPK: ' + str(TopK))
            f.write('\n')
            f.write("true links : %.2f%%" % (count / len(links) * 100))
            f.write('\n')


def get_links(vec, ids1, ids2):
    similarityMatrix = vec[ids1].mm(vec[ids2].t())
    rank = get_topk_indices(similarityMatrix, TopK * 100).detach().cpu().numpy().tolist()
    links = set()
    used_inds = []
    count = 0
    for ill in rank:
        id1 = ids1[ill[0]]
        id2 = ids2[ill[1]]
        if id1 in used_inds or id2 in used_inds: continue
        used_inds.append(id1)
        used_inds.append(id2)
        links.add((id1, id2))
        count += 1
        if count == TopK:
            break
    return links


def get_two(vec1, vec2, ids1, ids2):
    k = 1
    similarityMatrix1 = vec1[ids1].mm(vec1[ids2].t())
    _, r1 = similarityMatrix1.topk(k)
    similarityMatrix2 = vec2[ids1].mm(vec2[ids2].t())
    _, r2 = similarityMatrix2.topk(k)
    link = []
    print(r2.size())
    for i in range(len(r1)):
        for j in r1[i]:
            if j in r2[i]:
                link.append((ids1[i], ids2[j]))
                break
    return link
