import pickle
import numpy as np
from tools import *


def load_ill(path):
    tups = []
    with open(path, "r") as file:
        for line in file:
            try:
                params = line.strip("\n").split("\t")
                tups.append(tuple([int(x) for x in params]))
            except:
                print(line)
    return tups


def load_ent(file_path):
    id2object = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        print('loading a (id2object)file...  ' + file_path)
        for line in f:
            th = line.strip('\n').split('\t')
            id2object[int(th[0])] = th[1]
    return id2object


def load_label(file_path):
    id2object = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        print('loading a (id2object)file...  ' + file_path)
        for line in f:
            th = line.strip('\n').split('\t')
            id2object[th[0]] = th[1]
    return id2object


def dicToTxt(dic: dict, path):
    with open(path, 'w', encoding='utf-8') as f:
        for key, value in dic.items():
            f.write(key + '\t' + str(value))
            f.write('\n')


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


def getDescription():
    label = load_label(LABEL_1)
    label.update(load_label(LABEL_2))

    with open(description_path, "rb") as f:
        description = pickle.load(f)
    ents = load_ent(DataPath + 'ent_ids_1')
    ents.update(load_ent(DataPath + 'ent_ids_2'))
    descriptionLists = []
    print(len(label))
    print(len(ents))
    for i in range(len(ents)):
        url = ents[i]
        if ISNAME:
            if label.get(url):
                descriptionLists.append(label[url])
        else:
            if description.get(url) or label.get(url):
                descriptionLists.append(description.get(url, label[url]))
    return descriptionLists


if __name__ == '__main__':
    with open(description_path, "rb") as f:
        description = pickle.load(f)
    ents = load_ent(DataPath + 'ent_ids_1')
    ents.update(load_ent(DataPath + 'ent_ids_2'))
    query = [9869, 20369]
    for i in query:
        url = ents[i]
        d = description.get(url)
        print(d)
