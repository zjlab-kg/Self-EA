from transformers import BertTokenizer, BertModel
from transformers import logging
from model.glow import *
from loadData import *
from transformers import BertModel
import torch
import torch.nn as nn
from Param import *

class Basic_Bert_Unit_model(nn.Module):
    def __init__(self, input_size, result_size):
        super(Basic_Bert_Unit_model, self).__init__()
        self.result_size = result_size
        self.input_size = input_size
        self.bert_model = BertModel.from_pretrained(model_file)
        self.out_linear_layer = nn.Linear(self.input_size, self.result_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, batch_word_list, attention_mask):
        x = self.bert_model(input_ids=batch_word_list, attention_mask=attention_mask)  # token_type_ids =token_type_ids
        sequence_output = x.last_hidden_state
        cls_vec = sequence_output[:, 0]
        output = self.dropout(cls_vec)
        output = self.out_linear_layer(output)
        return output
BASIC_BERT_UNIT_MODEL_SAVE_PATH = "../Save_model/"
BASIC_BERT_UNIT_MODEL_SAVE_PREFIX = "DBP15K_{}en".format('ja')
MODEL_INPUT_DIM = 768
MODEL_OUTPUT_DIM = 400  # dimension of basic bert unit output embedding
def getDescriptionVector():
    logging.set_verbosity_warning()
    tokenizer = BertTokenizer.from_pretrained(model_file)

    bert_model_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX + "model" + '.p'
    Model = Basic_Bert_Unit_model(MODEL_INPUT_DIM, MODEL_OUTPUT_DIM)
    Model.load_state_dict(torch.load(bert_model_path, map_location='cpu'))
    print("loading basic bert unit model from:  {}".format(bert_model_path))
    Model.eval()
    for name, v in Model.named_parameters():
        v.requires_grad = False
    Model = Model.cuda(CUDA)

    descriptionLists = getDescription()
    vectors = []
    token = tokenizer(descriptionLists, max_length=200, truncation=True, padding=True)
    with torch.no_grad():
        for i in range(0, len(descriptionLists), batchSiza):
            attentionMask = torch.tensor(token['attention_mask'][i:i + batchSiza]).cuda(CUDA)
            input_ids = torch.tensor(token['input_ids'][i:i + batchSiza]).cuda(CUDA)
            z = Model(input_ids, attentionMask)  # 输出为[隐藏层数量, batchSize, ]
            vectors.extend(z.detach().cpu().numpy())
    vectors = np.array(vectors)
    with open('../Save_model/supDesVector', 'wb')as f:
        np.save(f, vectors)


if __name__ == '__main__':
    getDescriptionVector()
