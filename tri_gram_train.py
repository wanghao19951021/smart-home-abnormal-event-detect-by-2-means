# ngram模型的训练代码
# 输入原始日志的名称;event_result.log
# 输出模型mini_tri_gram.pkl

#
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

import json
import sys
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime as dt

CONTEXT_SIZE = 2
EMBEDDING_DIM = 100

# 获取日志名称
log_name = sys.argv[-1]
col_names = ['date', 'time', 'sensor_id', 'val']

homeass_df = pd.read_table(log_name, sep="\s+", names=col_names)

all_sensor_id_list = homeass_df.sensor_id.unique().tolist()

# 格式化处理一些事件的状态，我们使用 0 代表关闭的状态， 1 代表打开的状态

homeass_df = homeass_df.applymap(lambda x: '1' if (x == 'on' or x == 'OPEN') else x)
homeass_df = homeass_df.applymap(lambda x: '0' if (x == 'off' or x == 'CLOSE') else x)

homeass_df = homeass_df.applymap(lambda x: '1' if (x == 'ON' or x == 'open') else x)
homeass_df = homeass_df.applymap(lambda x: '0' if (x == 'OFF' or x == 'close') else x)

# 时间信息的格式 进行一些处理
temp_date_time = homeass_df['date'] + " " + homeass_df['time']

for i in tqdm(range(temp_date_time.shape[0])):
    if len(temp_date_time.iloc[i].split(':')[-1].split('.')[0]) > 2:
        size_1 = temp_date_time.iloc[i].rfind(':')
        size_2 = temp_date_time.iloc[i].find('.')
        temp_date_time.iloc[i] = temp_date_time.iloc[i][0:size_1] + temp_date_time.iloc[i][size_1:size_2 - 1] + \
                                 temp_date_time.iloc[i][size_2:-1]
        print(temp_date_time.iloc[i])

homeass_df['datetime'] = pd.to_datetime(temp_date_time)

del homeass_df['date']
del homeass_df['time']

# 显示出所有设备
target_list = ['T001', 'AC01', 'T002', 'T003', 'M024', 'L002', 'M025', 'M021',
               'L010', 'M019', 'M018', 'M017', 'M016', 'M015', 'L001', 'L008',
               'M014', 'M008', 'M009', 'M007', 'M006', 'M005', 'M011', 'M010',
               'M001', 'M023', 'M022', 'L003', 'M026', 'M027', 'M028', 'M029',
               'M037', 'L005', 'L007', 'M030', 'DOOR01', 'M036', 'M032', 'M033',
               'M031', 'M013', 'M035', 'M034', 'F002', 'L006', 'M002', 'M003',
               'L009', 'M004', 'M012', 'L004', 'F001', 'D007', 'I003', 'I007',
               'I009', 'I008', 'A001', 'A002', 'L011', 'A003', 'I002', 'I004',
               'I005', 'I006', 'I001', 'M038', 'M039', 'M040', 'M041', 'E001']

homeass_df['event_name'] = homeass_df['sensor_id'] + "_" + homeass_df['val']

# 统计并显示全部事件出现的次数, 共计3895条
event_count_series = homeass_df.event_name.value_counts()
event_count_series = event_count_series.sort_index()

target_sensor_id_list = homeass_df.sensor_id.unique()
b_target_sensor_id_list = [x for x in target_sensor_id_list if ("M0" in x or "D0" in x)]

homeass_df.event_name.value_counts().sort_index()

# 标注event
event2id = {event: i for i, event in enumerate(homeass_df.event_name.unique().tolist())}

train_homeass_df = homeass_df

event_id_list = train_homeass_df.event_name.tolist()
for i in range(len(event_id_list)):
    event_id_list[i] = event2id[event_id_list[i]]

train_homeass_df['event_id'] = pd.Series(event_id_list)

# train_homeass_df 稳定排序, 使得所有的相同时间戳事件按照被记录的顺序来排列
train_homeass_df.sort_values(by=['datetime'], ascending=[True], kind='mergesort', inplace=True)


# 按照固定的时间间隔，对事件的序列进行分割，并对分割后的每一个子序列进行检测
# second_num 为输入的秒钟数

def get_input_events_by_time(whole_df, seconds_num):
    time_span = dt.timedelta(seconds=seconds_num)

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

train_df_list = get_input_events_by_time(train_homeass_df, seconds_num=3600)


# 同时生成id对应的event
def dict_reverse(d):
    return dict([(v, k) for (k, v) in d.items()])


id2event = dict_reverse(event2id)

trigram = []
for input_df in tqdm(train_df_list):
    nrow = input_df.shape[0]

    sig_list = [
        ((str(input_df.iloc[i].event_name), str(input_df.iloc[i + 1].event_name)), str(input_df.iloc[i + 2].event_name))
        for i in range(nrow - 2)]
    trigram.extend(sig_list)


gc.collect()

vocb = set(event2id)


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


ngrammodel = event_NgramModel(len(event2id), CONTEXT_SIZE, EMBEDDING_DIM)
criterion = nn.NLLLoss()
optimizer = optim.SGD(ngrammodel.parameters(), lr=1e-3)


use_cuda = torch.cuda.is_available()  # 判断GPU是否存在可用
device = torch.device('cuda:0' if use_cuda else 'cpu')
ngrammodel.to(device)


for epoch in range(100):

    print('epoch: {}'.format(epoch + 1))
    print('*' * 10)

    running_loss = 0

    for data in trigram:
        word, label = data
        word = Variable(torch.LongTensor([event2id[i] for i in word])).to(device)
        label = Variable(torch.LongTensor([event2id[label]])).to(device)

        # 清零梯度
        ngrammodel.zero_grad()

        # forward
        out = ngrammodel(word)
        loss = criterion(out, label)
        running_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del word, label

    print('Loss: {:.6f}'.format(running_loss / len(event2id)))


# 保存
torch.save(ngrammodel, 'mini_tri_gram_model..pkl')
# 加载
ngrammodel = torch.load('mini_tri_gram_model.pkl')