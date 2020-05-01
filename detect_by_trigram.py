# ngram模型的异常检测代码
# 输入两个参数，第一个是ngram的模型 nimi_tri_gram_model.pkl
# 第二个是待测的包含异常事件的数据集  跳窗攻击./attacked_rub_house_from_window_big_new.csv 或系统异常./random_attack_big.csv
# 输出检测结果，将被判定位异常的事件元组输出到新文件中 suspicious_random_trigram.log 和 suspicious_window_trigram.log

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import *

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

import gc
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
from matplotlib.ticker import FuncFormatter

# 每一个事件有相应的id作为标识

with open('event2id.json', 'r') as f:
    event2id = json.load(f)


if sys.argv[-1] == './attacked_rub_house_from_window_big_new.csv':
    fp = open("suspicious_window_trigram.log", "w")
elif sys.argv[-1] == './random_attack_big.csv':
    fp = open("suspicious_random_trigram.log", "w")
else:
    print("Input Error!")


# 同时生成id对应的event
def dict_reverse(d):
    return dict([(v, k) for (k, v) in d.items()])


id2event = dict_reverse(event2id)


# 加载模型

class event_NgramModel(nn.Module):

    def __init__(self, vocb_size, context_size, n_dim):
        super(event_NgramModel, self).__init__()
        self.n_word = vocb_size
        self.embedding = nn.Embedding(self.n_word, n_dim)
        self.linear1 = nn.Linear(context_size * n_dim, 128)
        self.linear2 = nn.Linear(128, self.n_word)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1, -1)
        out = self.linear1(emb)
        out = F.relu(out)  # 激激活函数
        out = self.linear2(out)
        log_prob = F.log_softmax(out, dim=1)
        return log_prob


model_name = sys.argv[-2]
ngrammodel = torch.load(model_name)


# 读取异常检测测试的输入事件序列、并进行相应的处理
col_names = ['sensor_id', 'val', 'datetime', 'flag']
log_name = sys.argv[-1]
test_attack_1_df = pd.read_csv(log_name, sep=",", names=col_names)

# 数据处理
test_attack_1_df =  test_attack_1_df.drop([0], axis = 0)
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

# 按时间分割序列
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


# 计算一个元组的条件概率
def get_one_out_pro(event_1, event_2, event_3):
    word = torch.LongTensor([event_1, event_2])
    out_tensor = ngrammodel(word)
    out_array = out_tensor.detach().numpy()

    out_pro_by_log = out_array[0][event_3]

    return out_pro_by_log


# 输出给定序列的n-gram预测概率， 随机异常
def get_pro_and_flag_from_event_list_random(input_df, thresholds):
    nrow, ncol = input_df.shape

    pro_log_list = []
    flag_list = []
    for i in range(nrow - 2):
        event_id_1 = input_df.iloc[i].event_id
        event_id_2 = input_df.iloc[i + 1].event_id
        event_id_3 = input_df.iloc[i + 2].event_id

        event_id_3_pro_log = get_one_out_pro(event_id_1, event_id_2, event_id_3)
        event_id_3_flag = int(input_df.iloc[i + 2].flag)
        pro_log_list.append(event_id_3_pro_log)
        flag_list.append(event_id_3_flag)

        if event_id_3_pro_log < thresholds:
            fp.write("%s\t%s\t%s\t%f\n" % (id2event[event_id_1], id2event[event_id_2],
                                                 id2event[event_id_3], np.exp(event_id_3_pro_log)))

    return pro_log_list, flag_list


# 输出给定序列的n-gram预测概率， 跳窗抢劫，只检测第一个
def get_pro_and_flag_from_event_list_window(input_df, thresholds):
    nrow, ncol = input_df.shape

    pro_log_list = []
    flag_list = []
    for i in range(nrow - 2):

        event_id_1 = input_df.iloc[i].event_id
        event_id_2 = input_df.iloc[i + 1].event_id
        event_id_3 = input_df.iloc[i + 2].event_id

        event_id_3_pro_log = get_one_out_pro(event_id_1, event_id_2, event_id_3)
        event_id_3_flag = int(input_df.iloc[i + 2].flag)
        pro_log_list.append(event_id_3_pro_log)
        if event_id_3 == event2id['M012_1'] and int(input_df.iloc[i + 2].flag) == 1:
            flag_list.append(event_id_3_flag)
        else:
            flag_list.append(0)

        if event_id_3_pro_log < thresholds:
            fp.write("%s\t%s\t%s\t%f\n" % (id2event[event_id_1], id2event[event_id_2],
                                                 id2event[event_id_3], np.exp(event_id_3_pro_log)))

    return pro_log_list, flag_list


# 对给出的概率进行阈值判定
def to_pred(pro_log_list, q):
    pred_list = []
    for num in pro_log_list:

        if num < q:
            pred_list.append(1)
        else:
            pred_list.append(0)

    return pred_list


gc.collect()

# 对跳窗，只检测第一个异常出现，对系统随机异常则进行全部异常元组的检测
# 两种异常用不同的函数

pro_log_list = []
flag_list = []
pred_list = []
thresholds = 0 # 阈值

print("Now calculate probability for events! \n")
if len(input_df_list) == 1514: # 跳窗攻击

    thresholds = -6.9033136
    for df in tqdm(input_df_list):
        pro_log_list.extend(get_pro_and_flag_from_event_list_window(df, thresholds)[0])
        flag_list.extend(get_pro_and_flag_from_event_list_window(df, thresholds)[1])

    pred_list = to_pred(pro_log_list, thresholds)


elif len(input_df_list) == 1598: # 系统异常

    thresholds = -5.377418
    for df in tqdm(input_df_list):
        pro_log_list.extend(get_pro_and_flag_from_event_list_random(df, thresholds)[0])
        flag_list.extend(get_pro_and_flag_from_event_list_random(df, thresholds)[1])

    pred_list = to_pred(pro_log_list, thresholds)


p_score = precision_score(flag_list, pred_list)
r_score = recall_score(flag_list, pred_list)
f1 = f1_score(flag_list, pred_list, average='binary')
print("precision_score： ", p_score)
print("recall_score： ", r_score)
print("f1_score： ", f1)


fp.close()
print("End!!")

