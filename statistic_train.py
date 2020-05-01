# 关联权重图模型的训练代码
# 输入原始日志的名称;event_result.log
# 输出event_trans.npy


# 导入模块
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

# 读取文件
# 获取日志名称
log_name = sys.argv[-1]
col_names = ['date', 'time', 'sensor_id', 'val']

homeass_df = pd.read_table(log_name, sep="\s+", names=col_names)

# s数据处理

all_sensor_id_list = homeass_df.sensor_id.unique().tolist()

# 格式化处理一些事件的状态，我们使用 0 代表关闭的状态， 1 代表打开的状态

homeass_df = homeass_df.applymap(lambda x: '1' if (x=='on' or x=='OPEN') else x)
homeass_df = homeass_df.applymap(lambda x: '0' if (x=='off' or x=='CLOSE') else x)

homeass_df = homeass_df.applymap(lambda x: '1' if (x=='ON' or x=='open') else x)
homeass_df = homeass_df.applymap(lambda x: '0' if (x=='OFF' or x=='close') else x)

# 时间信息的格式 进行一些处理
temp_date_time = homeass_df['date'] + " " +  homeass_df['time']

for i in tqdm(range (temp_date_time.shape[0])):
    if len(temp_date_time.iloc[i].split(':')[-1].split('.')[0]) > 2:
        size_1 = temp_date_time.iloc[i].rfind(':')
        size_2 = temp_date_time.iloc[i].find('.')
        temp_date_time.iloc[i] = temp_date_time.iloc[i][0:size_1] + temp_date_time.iloc[i][size_1:size_2-1] + temp_date_time.iloc[i][size_2:-1]
        print( temp_date_time.iloc[i])


homeass_df['datetime'] = pd.to_datetime(temp_date_time )

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


homeass_df['event_name'] = homeass_df['sensor_id'] + "_" +  homeass_df['val']

# 统计并显示全部事件出现的次数, 共计3895条
event_count_series = homeass_df.event_name.value_counts()
event_count_series = event_count_series.sort_index()

target_sensor_id_list = homeass_df.sensor_id.unique()
b_target_sensor_id_list = [x  for x in target_sensor_id_list if ("M0" in x or "D0" in x)]

homeass_df.event_name.value_counts().sort_index()

# 标注event
event2id = {event:i for i, event in enumerate(homeass_df.event_name.unique().tolist())}


# 同时生成id对应的event
def dict_reverse(d):
    return dict([(v, k) for (k, v) in d.items()])


id2event = dict_reverse(event2id)

# 事件全部用于统计
train_homeass_df = homeass_df


def get_softmax(x_array):
    tmp = np.max(x_array)
    x_array -= tmp
    x_array = np.exp(x_array)
    tmp = np.sum(x_array)
    x_array /= tmp

    return x_array


# 定义函数将频数转化为概率
def get_simple_pro(x_array):
    sum_num = np.sum(x_array)

    x_array /= sum_num

    return x_array


event_id_list = train_homeass_df.event_name.tolist()
for i in range(len(event_id_list)):
    event_id_list[i] = event2id[event_id_list[i]]


train_homeass_df['event_id'] = pd.Series(event_id_list)

# train_homeass_df 稳定排序, 使得所有的相同时间戳事件按照被记录的顺序来排列
train_homeass_df.sort_values(by = ['datetime'] , ascending=[True], kind='mergesort', inplace=True)

# 生成概率转移矩阵的过程
event_trans = np.zeros((len(event2id), len(event2id)))

nrow = train_homeass_df.shape[0]

for i in tqdm(range(nrow - 1)):
    pre_id = train_homeass_df.iloc[i].event_id
    now_id = train_homeass_df.iloc[i + 1].event_id
    event_trans[pre_id][now_id] += 1

## 得到最终转移概率矩阵
for i in range(event_trans.shape[0]):
    event_trans[i] = get_simple_pro(event_trans[i])

# 存为numpy 是为了下次方面直接调用
np.save("event_trans_pro.npy", event_trans)
event_trans = np.load("event_trans_pro.npy")

