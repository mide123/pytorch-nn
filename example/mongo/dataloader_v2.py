# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2022/7/4 10:47
# Author     ：heyingjie
# Description：
"""
import numpy as np
import pandas as pd
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


def generate_train_data(data: pd.DataFrame, t='', features=[], label='', train_seq=[], test_seq=[],
                        window_size=7):
    data.sort_values([t], ascending=True, inplace=True, ignore_index=True)

    i = 0
    feature_value = data[features].values
    label_value = data[label].values
    train_size = len(feature_value)

    if i + window_size > train_size:
        padding_len = window_size - train_size
        padding_value = np.zeros((padding_len, feature_value.shape[1])) - 1
        feature_value = np.concatenate([padding_value, feature_value])
        train_seq.append((feature_value, np.array([label_value[-1]])))

        padding_len = window_size - train_size
        padding_value = np.zeros((padding_len, feature_value.shape[1])) - 1
        feature_value = np.concatenate([padding_value, feature_value[:len(feature_value)]])
        test_seq.append((feature_value, np.array([label_value[-1]])))

    while i + window_size <= train_size:

        if i + window_size == train_size:
            test_seq.append((feature_value[i: i + window_size], np.array([label_value[i + window_size]])))
        else:
            train_seq.append((feature_value[i: i + window_size], np.array([label_value[i + window_size]])))

        i += 1


def get_mongotv_dataloader(origin_data: pd.DataFrame, features=[], id_c="", label='cid_day_vv_t', window_size=7,
                           bash_size=64, need_test=True):
    train_seq = []
    test_seq = []

    origin_data.groupby([id_c]).apply(lambda x:
                                      generate_train_data(
                                          x,
                                          t='nth_day',
                                          features=features,
                                          label=label,
                                          train_seq=train_seq,
                                          test_seq=test_seq,
                                          window_size=window_size)
                                      )
    random.seed(46)
    if need_test:

        train = MyDataset(train_seq)
        test = MyDataset(test_seq)

        Dtr = DataLoader(dataset=train, batch_size=bash_size, shuffle=False, num_workers=0)
        Dte = DataLoader(dataset=test, batch_size=bash_size, shuffle=False, num_workers=0)

        return Dtr, Dte
    else:
        train_seq.extend(test_seq)
        train = MyDataset(train_seq)
        return DataLoader(dataset=train, batch_size=bash_size, shuffle=False, num_workers=0)
