from Param import *
import pickle
import numpy as np


def read_triple(file_path):
    tups = []
    with open(file_path, "r", encoding="utf-8") as fr:
        for line in fr:
            params = line.strip("\n").split("\t")
            tups.append(tuple([int(x) for x in params]))
    return tups


def read_idtuple_file(file_pathList):
    ret = []
    for file_path in file_pathList:
        with open(file_path, "r", encoding='utf-8') as f:
            for line in f:
                th = line.strip('\n').split('\t')
                x = []
                for i in range(len(th)):
                    x.append(int(th[i]))
                ret.append(tuple(x))
    return ret


if __name__ == '__main__':
    bert_model_other_data_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX + 'other_data.pkl'
    train_ill, test_ill, eid2data = pickle.load(open(bert_model_other_data_path, "rb"))
    triples = read_idtuple_file(["../data/dbp15k/zh_en/triples_1", "../data/dbp15k/zh_en/triples_2"])
    print(len(triples))
    print(len(eid2data))
    other_data = [train_ill, test_ill, eid2data, triples]
    pickle.dump(other_data, open(bert_model_other_data_path, "wb"))
