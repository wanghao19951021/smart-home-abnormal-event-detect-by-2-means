# smart-home-abnormal-event-detect-by-2-means
两种方法对智能家居进行事件关联建模和异常检测

执行方法：
对于第一种概率统计的检测方案：
1 python statistic_train.py event_result.log	
对原始数据进行统计，输出 event_trans.npy
2 python detect_by_weight.py event_trans_pro.npy attacked_rub_house_from_window_big_new.csv 
  python detect_by_weight.py event_trans_pro.npy random_attack_big.csv
# 输入两个参数，第一个是转移矩阵 event_trans_pro.npy
# 第二个是待测的包含异常事件的数据集  跳窗攻击./attacked_rub_house_from_window_big_new.csv 或系统异常./random_attack_big.csv
# 输出检测结果，将被判定位异常的事件元组输出到新文件中 suspicious_window_weight.log 和 suspicious_random_weight.log 



对第二种基于ngram的检测方案：

1 python tri_gram_train.py event_result.log
对原始数据进行ngram深度学习训练，生成mini_tri_gram.pkl，作为训练好的模型

2 python detect_by_trigram.py mini_tri_gram_model.pkl attacked_rub_house_from_window_big_new.csv 
  python detect_by_trigram.py mini_tri_gram_model.pkl random_attack_big.csv
# 输入两个参数，第一个是ngram的模型 mini_tri_gram_model.pkl
# 第二个是待测的包含异常事件的数据集  跳窗攻击./attacked_rub_house_from_window_big_new.csv 或系统异常./random_attack_big.csv
# 输出检测结果，将被判定位异常的事件元组输出到新文件中 suspicious_random_trigram.log 和 suspicious_window_trigram.log


