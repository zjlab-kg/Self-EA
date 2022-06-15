#!/usr/bin/python
cd Knowledge_Graph_Embedding/
python -u main.py --UNSUPK 6000
cd ../Alignment_Interaction/
python -u clean_attribute_data.py
python -u get_entity_embedding.py
python -u get_attributeValue_embedding.py
python -u get_neighView_and_desView_interaction_feature.py
python -u get_attributeView_interaction_feature.py
python -u get_hierarchy_feature.py
python -u interaction_model.py


