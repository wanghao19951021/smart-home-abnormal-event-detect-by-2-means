# 关联权重模型的异常检测代码
# 输入两个参数，第一个是转移矩阵 event_trans_pro.npy
# 第二个是待测的包含异常事件的数据集  跳窗攻击./attacked_rub_house_from_window_big_new.csv 或系统异常./random_attack_big.csv
# 输出检测结果，将被判定位异常的事件元组输出到新文件中 suspicious_window_weight.log 和 suspicious_random_weight.log


import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from sklearn.preprocessing import MinMaxScaler
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import *

# 导入模型event_trans_pro
model_name = sys.argv[-2]
event_trans = np.load(model_name)


if sys.argv[-1] == './attacked_rub_house_from_window_big_new.csv':
    fp = open("suspicious_window_weight.log", "w")
elif sys.argv[-1] == './random_attack_big.csv':
    fp = open("suspicious_random_weight.log", "w")
else:
    print("Input Error!")


# 每一个事件有相应的id作为标识

with open('event2id.json', 'r') as f:
    event2id = json.load(f)


# 同时生成id对应的event
def dict_reverse(d):
    return dict([(v, k) for (k, v) in d.items()])


id2event = dict_reverse(event2id)


# 这个是用来将概率转换为权重的函数
def g_sigmoid(x):
    return 1.0 / (1 + np.exp(5000 * (-x + 0.00021101498206372652))) - 0.0


# 输入一个序列， 我们返回通过概率转移矩阵计算得到的权重
def get_multi_event_pro_2_w(input_event_list, event_trans):
    n = len(input_event_list)
    if n < 1:
        return 1  # 序列中至少有两个事件

    # res_pro = g_sigmoid( event_single[input_event_list[0]] )
    res_pro = 1

    for i in range(n - 1):
        res_pro *= g_sigmoid(event_trans[input_event_list[i]][input_event_list[i + 1]])

    return res_pro

# 读取异常检测测试的输入事件序列、并进行相应的处理


col_names = ['sensor_id', 'val', 'datetime', 'flag']
log_name = sys.argv[-1]
test_attack_1_df = pd.read_csv(log_name, sep=",", names=col_names)
test_attack_1_df = test_attack_1_df.drop([0], axis = 0)
test_attack_1_df['event_name'] = test_attack_1_df['sensor_id'] + '_' + test_attack_1_df['val']

nrow = test_attack_1_df.shape[0]
event_id_list = []
new_e_list = []
for i in tqdm(range(nrow)):

    try:
        event_id_list.append(event2id[test_attack_1_df.iloc[i]['event_name']])
    except KeyError:
        event_id_list.append(-1)
        new_e_list.append(test_attack_1_df.iloc[i]['event_name'])


test_attack_1_df = test_attack_1_df.reset_index(drop=True)
test_attack_1_df['event_id'] = pd.Series(event_id_list)


test_attack_1_df = test_attack_1_df.drop(['sensor_id', 'val', 'event_name'], axis = 1)
the_event_id_list = test_attack_1_df.event_id.tolist()


time_span = dt.timedelta(seconds=3600)
test_attack_1_df.datetime = pd.to_datetime(test_attack_1_df.datetime)


def get_input_events_by_hour(whole_df):

    input_df_list = []

    nrow, ncol = whole_df.shape

    start_time = whole_df.iloc[0].datetime
    temp_time = start_time
    k = 0

    for i in tqdm(range(nrow)):

        temp_time = whole_df.iloc[i].datetime

        if (start_time + time_span) <= temp_time:
            df_input = whole_df[k:i]
            input_df_list.append(df_input)
            k = i
            start_time = temp_time

    df_input = whole_df[k:]
    input_df_list.append(df_input)

    return input_df_list


# 分割事件序列
input_df_list = get_input_events_by_hour(test_attack_1_df)


# 这里重新定义了sigmoid函数、我们需要的是对概率为0的转移事件输出0的权重
def sigmoid(x):
    if x <= 0:

        return 0
    else:

        return 1.0 / (1 + np.exp(5000 * (-x + 0.00021101498206372652))) - 0


# 简单事件两两概率预测当前事件触发概率
def get_simple_weight_from_event(input_df, event_trans):
    nrow, ncol = input_df.shape
    event_act_pro = 1

    for i in range(nrow):

        if i == 0:

            # 　print (event_single[the_event_id] )
            # event_act_pro *= sigmoid(event_single[the_event_id])
            continue

        else:

            pre_event_id = input_df.iloc[i - 1].event_id
            the_event_id = input_df.iloc[i].event_id

            if sigmoid(event_trans[pre_event_id][the_event_id]) == 0:
                fp.write("%s\t%s\n" % (id2event[pre_event_id], id2event[the_event_id]) )
                # print(pre_event_id, pre_event_id)

            if pre_event_id == -1 or the_event_id == -1:
                event_act_pro *= sigmoid(0)
            else:
                event_act_pro *= sigmoid(event_trans[pre_event_id][the_event_id])

            # print (event_act_pro)

    return event_act_pro


# 当出现转移概率为0的事件时，显示出来，它就是导致事件序列被检测为异常的原因
print("Now calculate weight for events\n")
event_act_weight_list = []

for input_df in input_df_list:
    event_act_weight_list.append(get_simple_weight_from_event(input_df, event_trans))

y_pred = []
for weight_out in event_act_weight_list:

    if weight_out <= 0.000:
        y_pred.append(1)
    else:
        y_pred.append(0)


def get_true_labels(input_df_list):

    y_true = []

    for i in tqdm(range(len(input_df_list))):

        temp_df = input_df_list[i]
        t_flag = 0
        for j in range(temp_df.shape[0]):
            if temp_df.iloc[j].flag == 1:
                t_flag = 1
                break

        y_true.append(t_flag)

    return y_true


print("Now get true labels !!\n")
y_true = get_true_labels(input_df_list )
p_score = precision_score(y_true, y_pred)
r_score = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='binary')

print("precision_score： ", p_score)
print("recall_score： ", r_score)
print("f1_score： ", f1)

fp.close()
print("End!!")