import clean_attribute_data
import get_entity_embedding
import get_attributeValue_embedding
import get_neighView_and_desView_interaction_feature
import get_attributeView_interaction_feature
import get_hierarchy_feature
import interaction_model
from Param import *



def saveLog():
    with open(ResultFile, 'a') as f:
        f.write('GCN : ' + str(GCNLayer))
        f.write('\n')


if __name__ == '__main__':
    saveLog()
    clean_attribute_data.main()
    get_entity_embedding.main()
    get_attributeValue_embedding.main()
    get_neighView_and_desView_interaction_feature.main()
    get_attributeView_interaction_feature.main()
    get_hierarchy_feature.main()
    interaction_model.main()


