import json

id_url_res_file_path = '../GCN_interaction_model/output/case_analysis.json'

has_img_id_file_path = '../SeedCollection/output/ent_keys.txt'

with open(id_url_res_file_path, 'r') as file:
    data = json.load(file)

with open(has_img_id_file_path, 'r') as file:
    has_img_ids = file.read().split('\n')

case_analysis = []

has_img_ids_int = []
for c in has_img_ids:
    try:
        has_img_ids_int.append(int(c))
    except:
        pass

has_img_ids = has_img_ids_int

for item in data:
    case_analysis_item = {}
    case_analysis_item['e1_url'] = item[0]
    case_analysis_item['e2_pred_url'] = item[1]
    case_analysis_item['e2_true_url'] = item[2]
    case_analysis_item['correct'] = item[3]
    case_analysis_item['e1_id'] = item[4]
    case_analysis_item['e2_pred_id'] = item[5]
    #if e1_id or e2_id has image
    if item[4] in has_img_ids:
        case_analysis_item['e1_has_img'] = True
    else:
        case_analysis_item['e1_has_img'] = False
    if item[5] in has_img_ids:
        case_analysis_item['e2_has_img'] = True
    else:
        case_analysis_item['e2_has_img'] = False

    case_analysis.append(case_analysis_item)

print(len(case_analysis))

#save to file
save_path = 'case_analysis_processed.json'
with open(save_path, 'w') as file:
    json.dump(case_analysis, file, indent=4)

