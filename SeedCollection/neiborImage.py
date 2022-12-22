import time
import numpy as np
from tools import *

ENTITY_NEIGH_MAX_NUM = 50  # max sampling neighbor num of entity
KERNEL_NUM = 1
CANDIDATE_NUM = 10


def getNeighborFeature(ent_emb, rel_triples, entity_pairs):
    ent_pad_id = len(ent_emb)
    dim = len(ent_emb[0])
    ent_emb.append([0.0 for _ in range(dim)])  # <PAD> embedding
    neigh_dict = neigh_ent_dict_gene(rel_triples, max_length=ENTITY_NEIGH_MAX_NUM,
                                     pad_id=ent_pad_id)
    # generate neighbor-view interaction feature
    neighViewInter = neighborView_interaction_F_gene(entity_pairs, ent_emb, neigh_dict, ent_pad_id, cuda_num=CUDA,
                                                     batch_size=2048)
    return neighViewInter


# 獲取候選實體對
def getPair(ill, ent_emb):
    train_ids_1 = [e1 for e1, e2 in ill]
    train_ids_2 = [e2 for e1, e2 in ill]
    candidates = candidate_generate(train_ids_1, train_ids_2, ent_emb, CANDIDATE_NUM, bs=2048,
                                    cuda_num=CUDA)
    entity_pairs = all_entity_pairs_gene(candidates)
    turelink = 0
    for (h, r) in entity_pairs:
        if (h, r) in ill:
            turelink += 1
    print("tureNum:{} testNum:{} rate:{}".format(turelink, len(ill), turelink / len(ill)))
    return entity_pairs


def all_entity_pairs_gene(candidate_dict):
    # generate list of all candidate entity pairs.
    entity_pairs_list = []
    for e1 in candidate_dict.keys():
        for e2 in candidate_dict[e1]:
            entity_pairs_list.append((e1, e2))
    entity_pairs_list = list(set(entity_pairs_list))
    print("entity_pair (e1,e2) num is: {}".format(len(entity_pairs_list)))
    return entity_pairs_list


def candidate_generate(ents1, ents2, ent_emb, candidate_topk=50, bs=32, cuda_num=0):
    """
    return a dict, key = entity, value = candidates (likely to be aligned entities)
    """
    emb1 = np.array(ent_emb)[ents1].tolist()
    emb2 = np.array(ent_emb)[ents2].tolist()
    print("Test(get candidate) embedding shape:", np.array(emb1).shape, np.array(emb2).shape)
    print("get candidate by cosine similartity.")
    res_mat = cos_sim_mat_generate(emb1, emb2, bs, cuda_num=cuda_num)

    score, index = batch_topk(res_mat, bs, candidate_topk, largest=True, cuda_num=cuda_num)
    ent2candidates = dict()
    for i in range(len(index)):
        e1 = ents1[i]
        e2_list = np.array(ents2)[index[i]].tolist()
        ent2candidates[e1] = e2_list
    return ent2candidates


def kernel_mus(n_kernels):
    l_mu = [1]
    if n_kernels == 1:
        return torch.FloatTensor(l_mu)
    bin_size = 1.0 / (n_kernels - 1)  # score range from [0, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return torch.FloatTensor(l_mu)


def kernel_sigmas(n_kernels):
    l_sigma = [0.001]  # for exact match.
    # small variance -> exact match
    if n_kernels == 1:
        return torch.FloatTensor(l_sigma)
    l_sigma += [0.1] * (n_kernels - 1)
    return torch.FloatTensor(l_sigma)


def neigh_ent_dict_gene(rel_triples, max_length, pad_id=None):
    """
    get one hop neighbor of entity
    return a dict, key = entity, value = (padding) neighbors of entity
    """
    neigh_ent_dict = dict()
    for h, r, t in rel_triples:
        if h not in neigh_ent_dict:
            neigh_ent_dict[h] = []
        if t not in neigh_ent_dict:
            neigh_ent_dict[t] = []
    for h, r, t in rel_triples:
        if h == t:
            continue
        neigh_ent_dict[h].append(t)
        neigh_ent_dict[t].append(h)
    # In order to get the maximum number of neighbors randomly for each entity
    for e in neigh_ent_dict.keys():
        np.random.shuffle(neigh_ent_dict[e])
        np.random.shuffle(neigh_ent_dict[e])
        np.random.shuffle(neigh_ent_dict[e])
    for e in neigh_ent_dict.keys():
        neigh_ent_dict[e] = neigh_ent_dict[e][:max_length]
    if pad_id is not None:
        for e in neigh_ent_dict.keys():
            pad_list = [pad_id] * (max_length - len(neigh_ent_dict[e]))
            neigh_ent_dict[e] = neigh_ent_dict[e] + pad_list
    return neigh_ent_dict


def batch_dual_aggregation_feature_gene(batch_sim_matrix, attn_ne1, attn_ne2):
    """
    Dual Aggregation.
    [similarity matrix -> feature]
    :param batch_sim_matrix: [B,ne1,ne2]
    :param mus: [1,1,k(kernel_num)]
    :param sigmas: [1,1,k]
    :param attn_ne1: [B,ne1,1]
    :param attn_ne2: [B,ne2,1]
    :return feature: [B,kernel_num * 2].
    """
    sim_maxpooing_1, _ = batch_sim_matrix.topk(k=1, dim=-1)  # [B,ne1,1] #get max value.
    pooling_value_1 = sim_maxpooing_1
    log_pooling_sum_1 = torch.clamp(pooling_value_1, min=1e-10)* attn_ne1  # [B,ne1,k]
    log_pooling_sum_1 = torch.sum(log_pooling_sum_1, 1)  # [B,k]

    sim_maxpooing_2, _ = torch.transpose(batch_sim_matrix, 1, 2).topk(k=1, dim=-1)  # [B,ne2,1]
    pooling_value_2 = sim_maxpooing_2
    log_pooling_sum_2 = torch.clamp(pooling_value_2, min=1e-10) * attn_ne2  # [B,ne2,k]
    log_pooling_sum_2 = torch.sum(log_pooling_sum_2, 1)  # [B,k]

    batch_ne2_num = attn_ne2.sum(dim=1)  # [B,1]
    batch_ne2_num = torch.clamp(batch_ne2_num, min=1e-10)
    log_pooling_sum_2 = log_pooling_sum_2 * (1 / batch_ne2_num)  # [B,k]

    batch_ne1_num = attn_ne1.sum(dim=1)  # [B,1]
    batch_ne1_num = torch.clamp(batch_ne1_num, min=1e-10)
    log_pooling_sum_1 = log_pooling_sum_1 * (1 / batch_ne1_num)  # [B,k]
    return log_pooling_sum_1+log_pooling_sum_2


def neighborView_interaction_F_gene(ent_pairs, ent_emb_list, neigh_dict, ent_pad_id, cuda_num=0, batch_size=512):
    """
    Neighbor-View Interaction.
    use Dual Aggregation and Neighbor-View Interaction to generate Similarity Feature between entity pairs.
    return entity pairs and features(between entity pairs)
    """
    start_time = time.time()
    e_emb = torch.FloatTensor(ent_emb_list).cuda(cuda_num)
    e_emb = F.normalize(e_emb, p=2, dim=-1)
    # print("sigmas:",sigmas)
    # print("mus:",mus)

    all_features = []
    # print("entity_embedding shape:",e_emb.shape)
    for start_pos in range(0, len(ent_pairs), batch_size):
        batch_ent_pairs = ent_pairs[start_pos: start_pos + batch_size]
        e1s = [e1 for e1, e2 in batch_ent_pairs]
        e2s = [e2 for e1, e2 in batch_ent_pairs]
        e1_tails = [neigh_dict[e1] for e1 in e1s]  # size: [B(Batchsize),ne1(e1_neighbor_max_num)]
        e2_tails = [neigh_dict[e2] for e2 in e2s]  # [B,ne2]
        e1_masks = np.ones(np.array(e1_tails).shape)
        e2_masks = np.ones(np.array(e2_tails).shape)
        e1_masks[np.array(e1_tails) == ent_pad_id] = 0
        e2_masks[np.array(e2_tails) == ent_pad_id] = 0
        e1_masks = torch.FloatTensor(e1_masks.tolist()).cuda(cuda_num).unsqueeze(-1)  # [B,ne1,1]
        e2_masks = torch.FloatTensor(e2_masks.tolist()).cuda(cuda_num).unsqueeze(-1)  # [B,ne2,1]
        e1_tails = torch.LongTensor(e1_tails).cuda(cuda_num)  # [B,ne1]
        e2_tails = torch.LongTensor(e2_tails).cuda(cuda_num)  # [B,ne2]
        e1_tail_emb = e_emb[e1_tails]  # [B,ne1,embedding_dim]
        e2_tail_emb = e_emb[e2_tails]  # [B,ne2,embedding_dim]
        sim_matrix = torch.bmm(e1_tail_emb, torch.transpose(e2_tail_emb, 1, 2))  # [B,ne1,ne2]
        features = batch_dual_aggregation_feature_gene(sim_matrix, e1_masks, e2_masks)
        features = features.detach().cpu().tolist()
        all_features.extend(features)

    print("all ent pair neighbor-view interaction features shape:", np.array(all_features).shape)
    print("get ent pair neighbor-view interaction features using time {:.3f}".format(time.time() - start_time))
    return all_features


def cos_sim_mat_generate(emb1, emb2, bs=128, cuda_num=0):
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


def batch_topk(mat, bs=128, topn=50, largest=False, cuda_num=0):
    # be equal to topk, Speed up computing with GPU
    res_score = []
    res_index = []
    axis_0 = mat.shape[0]
    for i in range(0, axis_0, bs):
        temp_div_mat = mat[i:min(i + bs, axis_0)].cuda(cuda_num)
        score_mat, index_mat = temp_div_mat.topk(topn, largest=largest)
        res_score.append(score_mat.cpu())
        res_index.append(index_mat.cpu())
    res_score = torch.cat(res_score, 0)
    res_index = torch.cat(res_index, 0)
    return res_score, res_index
