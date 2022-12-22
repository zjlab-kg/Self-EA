import sys
sys.path.append('..')
import GCN_basic_bert_unit.Param as params
ResultFile = params.ResultFile
CUDA = params.CUDA_NUM
Unsup = params.UNSUP
TopK = params.UNSUP_K

ISNAME = params.ISNAME
Translated = params.Translated
description_path = params.DES_DICT_PATH

batchSiza = 128
model_file = params.model_file

DataPath = params.DATA_PATH
files = [DataPath + 'ent_ids_1', DataPath + 'ent_ids_2']
description_feature_path = DataPath+"feature/"+('TD' if Translated else 'OD')
name_feature_path = DataPath+"feature/"+('TN' if Translated else 'ON')

illPath = params.Ill_Path

IMG_PATH = params.Image_feature_path
LABEL_1 = DataPath+('Tlabel_1' if Translated else 'label_1')
LABEL_2 = DataPath+'label_2'
