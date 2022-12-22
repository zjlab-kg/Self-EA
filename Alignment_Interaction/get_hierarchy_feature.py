import json
import pickle
from Param import *


def loadType(path):
    typeDic = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            k, v = line.strip('\n').split('\t')
            typeDic[int(k)] = v
    return typeDic


def loadJson(path):
    with open(path, 'r', encoding='utf-8') as f:
        class_dict = json.load(f)
    return class_dict


def getScore(type1, type2, cls):
    while type1 != type2:
        node1 = cls[type1]
        node2 = cls[type2]
        if node1['score'] > node2['score']:
            type1 = node1['fatherName']
        elif node1['score'] < node2['score']:
            type2 = node2['fatherName']
        else:
            type1 = node1['fatherName']
            type2 = node2['fatherName']
    return [cls[type2]['score']]


def main():
    print("start hierarchy feature:")
    entity_pairs = pickle.load(open(ENT_PAIRS_PATH, "rb"))
    type_dict = loadType(DATA_PATH + 'type')
    class_dict = loadJson(PATH + 'hierarchyTree.json')
    features = []
    for t, h in entity_pairs:
        score = getScore(type_dict[t], type_dict[h], class_dict)
        features.append(score)
    pickle.dump(features, open(HIERARCHY_FEATURE_PATH, "wb"))
    print("save hierarchy Feature in: ", HIERARCHY_FEATURE_PATH)


if __name__ == '__main__':
    main()
