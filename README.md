## Self-EA

This is code and datasets for Self-EA

The complete project code and data can be downloaded from here (https://drive.google.com/file/d/1KhbL0Ie2zW_PYv3TeO9AVEm1kUARhy_d/view?usp=sharing)

### Dependencies

- Python 3 (tested on 3.7.2)
- Pytorch (tested on 1.1.0)
- [transformers](https://github.com/huggingface/transformers) (tested on 2.1.1)
- Numpy

### How to Run

The model runs directly use:

```shell
bash run.sh
```

Note that `Param.py` is the config file.

### Dataset

**Description data**

- `data/dbp15k/2016-10-des_dict`: A dictionary storing entity descriptions, which can be loaded by pickle.load()

The description of the entity is extracted from **DBpedia**(<https://wiki.dbpedia.org/downloads-2016-10>)

**DBP15K**

Initial datasets are from **JAPE**(<https://github.com/nju-websoft/JAPE>).

There are three cross-lingual datasets in folder `data/dbp15k/` , take the dataset DBP15K(JA-EN) as an example, the folder `data/dbp15k/zh_en` contains:

- ent_ids_1: entity ids and entities in source KG (JA)

- ent_ids_2: entity ids and entities in target KG (EN)

- ref_pairs: entity links encoded by ids (Test Set)

- sup_pairs: entity links encoded by ids (Train Set)

- rel_ids_1: relation ids and relations in source KG (JA)

- rel_ids_2: relation ids and relations in target KG (EN)

- triples_1: relation triples encoded by ids in source KG (JA)

- triples_2: relation triples encoded by ids in target KG (EN)

- ja_att_triples: attribute triples of source KG (JA)

- en_att_triples: attribute triples of target KG (EN)

**Download additional files from BaiduYun Drive**
https://pan.baidu.com/s/1FiCg7Munb_WkXsnnb8B6XQ?pwd=4801

### Acknowledgement
Our codes are modified from **BERT-INT**(<https://github.com/kosugi11037/bert-int>). We appreciate the authors for making BERT-INT open-sourced.

