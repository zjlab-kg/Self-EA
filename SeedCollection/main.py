from transformers import BertTokenizer, BertModel
from transformers import logging
from model.glow import *
from loadData import *


def getDescriptionVector():
    logging.set_verbosity_warning()
    tokenizer = BertTokenizer.from_pretrained(model_file)
    model = BertModel.from_pretrained(model_file).cuda(CUDA)
    descriptionLists = getDescription()

    vectors = []
    token = tokenizer(descriptionLists, max_length=200, truncation=True, padding=True)
    with torch.no_grad():
        for i in range(0, len(descriptionLists), batchSiza):
            attentionMask = torch.tensor(token['attention_mask'][i:i + batchSiza]).cuda(CUDA)
            input_ids = torch.tensor(token['input_ids'][i:i + batchSiza]).cuda(CUDA)
            lengths = attentionMask.sum(dim=1, keepdim=True)
            hidden_state = model(input_ids, attentionMask, output_hidden_states=True,
                                 return_dict=True).hidden_states  # 输出为[隐藏层数量, batchSize, ]

            emb = (
                    (hidden_state[0] * attentionMask.unsqueeze(-1)).sum(dim=1) +
                    (hidden_state[-1] * attentionMask.unsqueeze(-1)).sum(dim=1)
            ).div(2 * lengths)  # (bsz, 789)
            glow = Glow(model.config.hidden_size).cuda(CUDA)
            z, loss = glow(emb)
            # z = emb
            vectors.extend(z.detach().cpu().numpy())
    vectors = np.array(vectors)
    with open(description_feature_path, 'wb')as f:
        np.save(f, vectors)


if __name__ == '__main__':
    getDescriptionVector()
