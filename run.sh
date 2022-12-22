#!/usr/bin/python
if [ ! -e "log" ]; then
  mkdir "log"
fi

i="Test"
cd SeedCollection/
python -u calculateSim.py >> ../log/"$i".log 2>&1
cd ../
cd GCN_basic_bert_unit/
python -u main.py >> ../log/"$i".log 2>&1
cd ../GCN_interaction_model/
python -u clean_attribute_data.py >> ../log/"$i".log 2>&1
python -u get_entity_embedding.py >> ../log/"$i".log 2>&1
python -u get_attributeValue_embedding.py >> ../log/"$i".log 2>&1
python -u get_neighView_and_desView_interaction_feature.py >> ../log/"$i".log 2>&1
python -u get_attributeView_interaction_feature.py >> ../log/"$i".log 2>&1
python -u get_hierarchy_feature.py >> ../log/"$i".log 2>&1
python -u interaction_model.py >> ../log/"$i".log 2>&1
cd ..

#种子对齐数量
#for i in {100,500,1000,2000,3000,4000,5000,6000,7000,8000}
#do
#  cd SeedCollection/
#  python -u calculateSim.py --UNSUPK $i >> ../log/"$i".log 2>&1
#  cd ../
#  cd GCN_basic_bert_unit/
#  python -u main.py --UNSUPK $i >> ../log/"$i".log 2>&1
#  cd ../GCN_interaction_model/
#  python -u clean_attribute_data.py >> ../log/"$i".log 2>&1
#  python -u get_entity_embedding.py >> ../log/"$i".log 2>&1
#  python -u get_attributeValue_embedding.py >> ../log/"$i".log 2>&1
#  python -u get_neighView_and_desView_interaction_feature.py >> ../log/"$i".log 2>&1
#  python -u get_attributeView_interaction_feature.py >> ../log/"$i".log 2>&1
#  python -u get_hierarchy_feature.py >> ../log/"$i".log 2>&1
#  python -u interaction_model.py >> ../log/"$i".log 2>&1
#  cd ..
#done

# 计算kernel数量
#cd SeedCollection/
#python -u calculateSim.py --UNSUPK 6000 >> ../log/"6000".log 2>&1
#cd ../
#cd GCN_basic_bert_unit/
#python -u main.py --UNSUPK 6000 >> ../log/"6000".log 2>&1
#cd ../GCN_interaction_model/
#for i in {1,2,4,6,8,12,16,20,24}
#do
#  python -u main.py --KERNEL $i >> ../log/"$i".log 2>&1
#done


