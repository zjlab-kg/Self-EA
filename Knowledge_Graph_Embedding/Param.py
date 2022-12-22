import argparse
import sys

sys.path.append('..')

parse = argparse.ArgumentParser()
parse.add_argument("--UNSUPK", type=int, default=6000, required=False)
parse.add_argument("--LANG", type=str, default="fr", required=False)
parse.add_argument("--KERNEL", "-k", type=int, default=4, required=False)
parse.add_argument("--GCNLayer", "-gl", type=int, default=4, required=False)

args = parse.parse_args()
print("In params:")
LANG = args.LANG  # language 'zh'/'ja'/'fr'
EPOCH_NUM = 15  # training epoch num
CUDA_NUM = 7  # used GPU num

MODEL_INPUT_DIM = 768
MODEL_OUTPUT_DIM = 400  # dimension of basic bert unit output embedding
RANDOM_DIVIDE_ILL = False  # if is True: get train/test_ILLs by random divide all entity ILLs, else: get train/test ILLs from file.
TRAIN_ILL_RATE = 0.3  # (only work when RANDOM_DIVIDE_ILL == True) training data rate. Example: train ILL number: 15000 * 0.3 = 4500.

SEED_NUM = 11037

UNSUP_K = args.UNSUPK

DropOut = 0.1
UNITS = "400,300,100"
GCNLayer = args.GCNLayer

UNSUP = True
# 是否用名称翻译得到的种子对齐
TILL = False
# iterative
ITERATIVE = True
ITERA_EPOCH = 2
ITRA_TOPK = 1
ENFORCE = True
ADDNUM = 200
EARLY_STOP = False

NEAREST_SAMPLE_NUM = 128
CANDIDATE_GENERATOR_BATCH_SIZE = 128

TOPK = 50
NEG_NUM = 2  # negative sample num
MARGIN = 3  # margin
LEARNING_RATE = 5*1e-6  # learning rate
TRAIN_BATCH_SIZE = 25
TEST_BATCH_SIZE = 128

DES_LIMIT_LENGTH = 128  # max length of description/name.

DATA_PATH = r"../data/dbp15k/{}_en/".format(LANG)  # data path

Translated = False
ISNAME = False

DES_DICT_PATH = '../data/dbp15k/translated_description' if Translated else '../data/dbp15k/2016-10-des_dict'  # description data path
Image_feature_path = "../data/EAKpkls/" + LANG + "_en_GA_id_img_feature_dict.pkl"
MODEL_SAVE_PATH = "../Save_model/"  # model save path
MODEL_SAVE_PREFIX = "DBP15K_{}en".format(LANG)
Ill_Path = DATA_PATH + "/ill/" + ("name_" if ISNAME else "description_") + str(UNSUP_K)

import os

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

print("CUDA:", CUDA_NUM)

print("NEG_NUM:", NEG_NUM)
print("MARGIN:", MARGIN)
print("LEARNING RATE:", LEARNING_RATE)
print("TRAIN_BATCH_SIZE:", TRAIN_BATCH_SIZE)
print("TEST_BATCH_SIZE", TEST_BATCH_SIZE)
print("DES_LIMIT_LENGTH:", DES_LIMIT_LENGTH)
print("RANDOM_DIVIDE_ILL:", RANDOM_DIVIDE_ILL)
print("UNDUPK:", UNSUP_K)
print("")
print("")

model_file = '../transformers/LaBSE'
# model_file = '../transformers/BERT'
ResultFile = '../log/result'
