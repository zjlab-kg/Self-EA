import neiborImage
from transformers import logging
from loadData import *


def getLink():
    logging.set_verbosity_warning()
    ents = load_ent(DataPath + 'ent_ids_1')
    ents2 = load_ent(DataPath + 'ent_ids_2')
    ids1 = list(ents.keys())
    ids2 = list(ents2.keys())
    ents.update(ents2)
    ills = load_ill(DataPath + 'ill_ent_ids')
    with open(description_feature_path, 'rb')as f:
        description = np.load(f)
    with open(name_feature_path, 'rb')as f:
        naem = np.load(f)
    name = F.normalize(torch.Tensor(naem).cuda(CUDA))
    description = F.normalize(torch.Tensor(description).cuda(CUDA))
    img = load_img(len(ids1) + len(ids2), IMG_PATH)
    img = F.normalize(torch.Tensor(img).cuda(CUDA))

    vec = torch.cat([description * 0.2, img], dim=1)
    # vec = img
    # link = get_two(name, description, ids1, ids2)
    link = get_links(vec, ids1, ids2)
    calculate_correct_rate(link, ills)
    return link


def getNeighborSimilarity():
    logging.set_verbosity_warning()
    ents = load_ent(DataPath + 'ent_ids_1')
    ents2 = load_ent(DataPath + 'ent_ids_2')
    triples = load_ill(DataPath + 'triples_1')
    triple2 = load_ill(DataPath + 'triples_2')
    triples.extend(triple2)
    ids1 = list(ents.keys())
    ids2 = list(ents2.keys())
    ents.update(ents2)
    ills = load_ill(DataPath + 'ill_ent_ids')
    img = load_img(len(ids1) + len(ids2), IMG_PATH)
    img = F.normalize(torch.Tensor(img).cuda(CUDA))
    emb = img.detach().cpu().tolist()
    ent_pair = list(get_links(img, ids1, ids2))
    calculate_correct_rate(ent_pair, ills)

    neiSimilarity = neiborImage.getNeighborFeature(emb, triples, ent_pair)
    neiSimilarity = np.array(neiSimilarity)
    with open("img.np", 'wb') as f:
        np.save(f, neiSimilarity)


def res():
    logging.set_verbosity_warning()
    ents = load_ent(DataPath + 'ent_ids_1')
    ents2 = load_ent(DataPath + 'ent_ids_2')
    triples = load_ill(DataPath + 'triples_1')
    triple2 = load_ill(DataPath + 'triples_2')
    triples.extend(triple2)
    ids1 = list(ents.keys())
    ids2 = list(ents2.keys())
    ents.update(ents2)
    ills = load_ill(DataPath + 'ill_ent_ids')
    img = load_img(len(ids1) + len(ids2), IMG_PATH)
    img = F.normalize(torch.Tensor(img).cuda(CUDA))
    emb = torch.mm(img, img.T)
    ent_pair = list(get_links(img, ids1, ids2))
    ifea = []
    for i, j in ent_pair:
        ifea.append(emb[i][j])
    with open("img.np", 'rb') as f:
        neiSimilarity = np.load(f)
    neiSimilarity = np.array()
    neiSimilarity = torch.tensor(neiSimilarity).view(-1)
    print(neiSimilarity.size())
    value, rank = neiSimilarity.topk(6000)
    link = set()
    used_inds = []
    count = 0
    for i in rank:
        id1 = ent_pair[i][0]
        id2 = ent_pair[i][1]
        if id1 in used_inds or id2 in used_inds: continue
        used_inds.append(id1)
        used_inds.append(id2)
        link.add((id1, id2))
        count += 1
        if count == 4000:
            break
    calculate_correct_rate(ent_pair, ills)
    calculate_correct_rate(link, ills)


if __name__ == '__main__':
    link = getLink()
    filepath = illPath
    print(illPath)
    with open(filepath, 'w') as f:
        for a, b in link:
            f.write(str(a))
            f.write('\t')
            f.write(str(b))
            f.write('\n')
