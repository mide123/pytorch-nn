# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2022/5/23 16:55
# Author     ：heyingjie
# Description：
"""
import numpy as np
import pandas as pd
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random


class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def get_bg_dataloader(path, window_size=7, bash_size=64):
    print('获取训练数据:')

    data_iter = pd.read_csv(path, encoding='utf-8', chunksize=3000)
    positive_size = 0
    seq = []
    for data in data_iter:
        for charging_guid, current_series, risk_series in data[["charging_guid", "current_series", "risk_series"]] \
                .values:

            current_arr = np.array([float(x) for x in current_series[1:].split(",")])
            risk_arr = np.array([float(x) for x in risk_series[1:].split(",")])
            for i in range(len(risk_arr) - window_size):
                if risk_arr[i + window_size] >= 0:
                    train_seq = torch.FloatTensor(current_arr[i: i + window_size])
                    train_label = torch.FloatTensor([risk_arr[i + window_size]]).view(-1)
                    if train_label == 1:
                        seq.append((train_seq, train_label))
                        positive_size = positive_size + 1
                    elif random.randint(1, 8) == 2:
                        seq.append((train_seq, train_label))
        # break
    print(f"总样本数：{len(seq)}, 正样本数：{positive_size}, 正样本比例：{positive_size / len(seq)}")

    # random.shuffle(seq)
    Dtr = seq[0:int(len(seq) * 0.7)]
    random.shuffle(Dtr)  # 避免过拟合，在此处进行shuffle

    Dte = seq[int(len(seq) * 0.7):len(seq)]

    random.shuffle(Dte)
    train_len = int(len(Dtr) / bash_size) * bash_size
    test_len = int(len(Dte) / bash_size) * bash_size

    Dtr, Dte = Dtr[:train_len], Dte[:test_len]

    train = MyDataset(Dtr)
    test = MyDataset(Dte)

    Dtr = DataLoader(dataset=train, batch_size=bash_size, shuffle=False, num_workers=0)
    Dte = DataLoader(dataset=test, batch_size=bash_size, shuffle=False, num_workers=0)

    return Dtr, Dte


def generate_train_data(data: pd.DataFrame, t='', features=[], label='', seq=[], window_size=7):
    data.sort_values([t], ascending=True, inplace=True, ignore_index=True)

    i = 0
    feature_value = data[features].values
    label_value = data[label].values
    train_size = len(label_value[label_value > 0])

    while i + window_size < train_size:
        seq.append((feature_value[i: i + window_size], np.array([label_value[i + window_size]])))
        i += 1


def get_mongotv_dataloader(origin_data: pd.DataFrame, features=[], id_c="", label='cid_day_vv_t', window_size=7,
                           bash_size=64, need_test=True):
    seq = []
    origin_data.groupby([id_c]).apply(lambda x:
                                      generate_train_data(
                                          x,
                                          t='nth_day',
                                          features=features,
                                          label=label,
                                          seq=seq,
                                          window_size=window_size)
                                      )
    random.seed(46)
    if need_test:

        random.shuffle(seq)
        Dtr = seq[0:int(len(seq) * 0.9)]
        Dte = seq[int(len(seq) * 0.9):len(seq)]

        train_len = int(len(Dtr) / bash_size) * bash_size
        test_len = int(len(Dte) / bash_size) * bash_size

        Dtr, Dte = Dtr[:train_len], Dte[:test_len]

        train = MyDataset(Dtr)
        test = MyDataset(Dte)

        Dtr = DataLoader(dataset=train, batch_size=bash_size, shuffle=False, num_workers=0)
        Dte = DataLoader(dataset=test, batch_size=bash_size, shuffle=False, num_workers=0)

        return Dtr, Dte
    else:
        train = MyDataset(seq)
        return DataLoader(dataset=train, batch_size=bash_size, shuffle=False, num_workers=0)


def ceshi_bg_dataloader():
    Dtr, Dte = get_bg_dataloader(path="E:/risk_train.csv")
    for x_train, y_train in Dtr:
        print(x_train.shape, y_train.shape)
        break


def ceshi_mongotv_dataloader():
    features = ['weekday', 'seriesNo', 'seriesId_t', 'channelId_t', 'cid_day_vv_t', 'is_holiday',
                'date_has_update']
    data = pd.read_csv("E:/data/mangguotv/rank_a_data.tsv", sep='\t')
    data['date_has_update'] = data['date_has_update'].apply(lambda x: 1 if x is True else 0)
    data['is_holiday'] = data['is_holiday'].apply(lambda x: 1 if x is True else 0)

    Dtr, Dte = get_mongotv_dataloader(
        origin_data=data,
        id_c='cid_t',
        label="cid_day_vv_t",
        features=features,
        window_size=10,
        bash_size=512
    )
    for seq, label in Dtr:
        print(seq.shape, label.shape)
        break


if __name__ == '__main__':
    ceshi_mongotv_dataloader()
