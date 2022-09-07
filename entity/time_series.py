# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2022/8/19 11:22
# Author     ：heyingjie
# Description：
"""
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def get_dataloader_mullticlass(data, features, label, window_size=7, bash_size=64, test_size=0.2, log=None):
    '''
    对于一行是一个序列的数据构建dataloader
    :param data:
    :param features:  [12,3,4,5,6,7,8]
    :param label:[1,1,1,2,2,2,2]
    :param window_size:
    :param bash_size:
    :param test_size:
    :param log:
    :return:
    '''
    log.debug('获取训练数据')
    random.seed(10)

    seq = []

    for current_series, risk_series in data[features + [label]] \
            .values:

        current_arr = np.array([float(x) for x in current_series[1:-1].split(",")])
        risk_arr = np.array([float(x) for x in risk_series[1:-1].split(",")])
        first_current = current_arr[0]
        first_risk = risk_arr[0]
        current_arr = np.hstack([np.array([first_current] * (window_size - 1)), current_arr])
        risk_arr = np.hstack([np.array([first_risk] * (window_size - 1)), risk_arr])
        for i in range(len(risk_arr) - window_size):
            if risk_arr[i + window_size] >= 0:
                train_seq_arr = current_arr[i: i + window_size]
                train_seq = torch.FloatTensor(train_seq_arr)
                train_label = torch.FloatTensor([risk_arr[i + window_size - 1]]).view(-1)
                seq.append((train_seq, train_label))

    # 先对数据进行划分，再进行shuffer
    train_size = 1 - test_size
    Dtr = seq[0:int(len(seq) * train_size)]
    random.shuffle(Dtr)  # 避免过拟合，在此处进行shuffle

    Dte = seq[int(len(seq) * train_size):len(seq)]

    random.shuffle(Dte)
    train_len = int(len(Dtr) / bash_size) * bash_size
    test_len = int(len(Dte) / bash_size) * bash_size

    Dtr, Dte = Dtr[:train_len], Dte[:test_len]

    train = MyDataset(Dtr)
    test = MyDataset(Dte)

    Dtr = DataLoader(dataset=train, batch_size=bash_size, shuffle=False, num_workers=0)
    Dte = DataLoader(dataset=test, batch_size=bash_size, shuffle=False, num_workers=0)

    return Dtr, Dte
