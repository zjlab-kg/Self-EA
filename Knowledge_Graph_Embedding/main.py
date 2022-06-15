import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW
import torch.optim as optim
import random
from eval_function import *
from Read_data_func import read_data
from Param import *
from Basic_Bert_Unit_model import Basic_Bert_Unit_model
from Batch_TrainData_Generator import Batch_TrainData_Generator
from train_func import train
import numpy as np
from GCN_mode import *

def fixed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_feature(Model, ent_ill, entid2data, batch_size, context=""):
    print("-----test start-----")
    start_time = time.time()
    print(context)
    Model.eval()
    with torch.no_grad():
        ents_1 = [e1 for e1, e2 in ent_ill]
        ents_2 = [e2 for e1, e2 in ent_ill]

        emb1 = []
        for i in range(0, len(ents_1), batch_size):
            batch_ents_1 = ents_1[i: i + batch_size]
            batch_emb_1 = entlist2emb(Model, batch_ents_1, entid2data, CUDA_NUM).detach().cpu().tolist()
            emb1.extend(batch_emb_1)
            del batch_emb_1

        emb2 = []
        for i in range(0, len(ents_2), batch_size):
            batch_ents_2 = ents_2[i: i + batch_size]
            batch_emb_2 = entlist2emb(Model, batch_ents_2, entid2data, CUDA_NUM).detach().cpu().tolist()
            emb2.extend(batch_emb_2)
            del batch_emb_2

        print("Cosine similarity of basic bert unit embedding res:")
        res_mat = cos_sim_mat_generate(emb1, emb2, batch_size, cuda_num=CUDA_NUM)
        score, top_index = batch_topk(res_mat, batch_size, topn=TOPK, largest=True, cuda_num=CUDA_NUM)
        hit_res(top_index)
    print("test using time: {:.3f}".format(time.time() - start_time))
    print("--------------------")


def entlist2emb(Model, entids, entid2data, cuda_num):
    """
    return basic bert unit output embedding of entities
    """
    batch_token_ids = []
    batch_mask_ids = []
    for eid in entids:
        temp_token_ids = entid2data[eid][0]
        temp_mask_ids = entid2data[eid][1]

        batch_token_ids.append(temp_token_ids)
        batch_mask_ids.append(temp_mask_ids)

    batch_token_ids = torch.LongTensor(batch_token_ids).cuda(cuda_num)
    batch_mask_ids = torch.FloatTensor(batch_mask_ids).cuda(cuda_num)

    batch_emb = Model(batch_token_ids, batch_mask_ids,entids)
    del batch_token_ids
    del batch_mask_ids
    return batch_emb


def main():
    # read data
    print("start load data....")
    ent_ill, train_ill, test_ill,index2rel, index2entity, rel2index, entity2index, ent2data, rel_triples_1, rel_triples_2, imgdata = read_data()
    print("---------------------------------------")
    print("all entity ILLs num:", len(ent_ill))
    print("rel num:", len(index2rel))
    print("ent num:", len(index2entity))
    print("triple1 num:", len(rel_triples_1))
    print("triple2 num:", len(rel_triples_2))
    print('tot ent:', len(ent2data))
    # model
    triples = copy.deepcopy(rel_triples_1)
    triples.extend(rel_triples_2)
    ENT_NUM = len(ent2data)

    # Model = Basic_Bert_Unit_model(MODEL_INPUT_DIM, MODEL_OUTPUT_DIM)
    Model = combine_model(ENT_NUM,triples)
    Model.cuda(CUDA_NUM)
    # if torch.cuda.device_count() >1:
    #     Model = nn.DataParallel(Model)

    # get train/test_ill
    if RANDOM_DIVIDE_ILL:
        # get train/test_ILLs by random divide all entity ILLs
        print("Random divide train/test ILLs!")
        random.shuffle(ent_ill)
        train_ill = random.sample(ent_ill, int(len(ent_ill) * TRAIN_ILL_RATE))
        test_ill = list(set(ent_ill) - set(train_ill))
        print("train ILL num: {}, test ILL num: {}".format(len(train_ill), len(test_ill)))
        print("train ILL | test ILL num:", len(set(train_ill) | set(test_ill)))
        print("train ILL & test ILL num:", len(set(train_ill) & set(test_ill)))
    else:
        # get train/test ILLs from file.
        print("get train/test ILLs from file \"sup_pairs\", \"ref_pairs\" !")
        print("train ILL num: {}, test ILL num: {}".format(len(train_ill), len(test_ill)))
        print("train ILL | test ILL:", len(set(train_ill) | set(test_ill)))
        print("train ILL & test ILL:", len(set(train_ill) & set(test_ill)))

    Criterion = nn.MarginRankingLoss(MARGIN, size_average=True)
    Optimizer = AdamW(Model.parameters(), lr=LEARNING_RATE)

    ent1 = [e1 for e1, e2 in ent_ill]
    ent2 = [e2 for e1, e2 in ent_ill]
    print(train_ill[:10])
    print(train_ill[-10:])

    print(len(train_ill))

    # training data generator(can generate batch-size training data)
    Train_gene = Batch_TrainData_Generator(train_ill, ent1, ent2, index2entity, batch_size=TRAIN_BATCH_SIZE,
                                           neg_num=NEG_NUM)
    train(Model, Criterion, Optimizer, Train_gene, train_ill, test_ill, ent2data, imgdata,triples)
    #
    # emb = []
    # ents = list(range(len(ent2data)))
    # for i in range(0, len(ent2data), TEST_BATCH_SIZE):
    #     batch_ents = ents[i: i + TEST_BATCH_SIZE]
    #     batch_emb = entlist2emb(Model, batch_ents, ent2data, CUDA_NUM).detach().cpu().tolist()
    #     emb.extend(batch_emb)
    #     del batch_emb
    # feature = np.array(emb)
    # print(feature.shape)
    # np.save('../output/bert.npy', feature, allow_pickle=True)


if __name__ == '__main__':
    main()
