# !/usr/bin/env python
# -*-coding:utf-8 -*-

import numpy as np
import random

import pandas as pd
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


def create_sequences(df, all_featuers, window_size, label, seq: list, step_size):

    current_arr = df[all_featuers]
    risk_arr = df[label]
    for i in range(len(risk_arr) - window_size - step_size):
        train_seq_arr = current_arr[i: i + window_size]
        train_seq = torch.FloatTensor(train_seq_arr.values)
        train_label = torch.FloatTensor(risk_arr.values[i + window_size + step_size])
        seq.append((train_seq, train_label))


def get_dataloader(seq_data, features=None, label=None, window_size=7, batch_size=64, test_size=0.2, step_size=0, shuffer=True):

    if label is None:
        label = []
    if features is None:
        features = []
    print('获取训练数据')
    seq = []
    create_sequences(df=seq_data, all_featuers=features, window_size=window_size, label=label, seq=seq, step_size=step_size)
    random.seed(10)

    # 先对数据进行划分，再进行shuffer
    train_len = int(len(seq) * (1-test_size))
    Dtr = seq[0: train_len]
    if shuffer is True:
        random.shuffle(Dtr)  # 避免过拟合，在此处进行shuffle
    Dte = seq[train_len:len(seq)]
    train = MyDataset(Dtr)
    test = MyDataset(Dte)
    Dtr = DataLoader(dataset=train, batch_size=batch_size, shuffle=False, num_workers=0)
    Dte = DataLoader(dataset=test, batch_size=batch_size, shuffle=False, num_workers=0)

    return Dtr, Dte

if __name__ == '__main__':
    data = pd.read_excel("../../data/工况3Mpa.xlsx")
    # 对数据进行归一化
    print(data.columns)
    features = ['进口压力（MPa）', '主油路出口压力(MPa)', '主油路流量（L/h）', '副油路出口压力（MPa）', '副油路流量（L/h）',
       '供油路流量', '主油路温度', '副油路温度']
    label = list(features)
    Dtr, Dte = get_dataloader(data, features=features, label=label)
    for x, y in Dtr:
        print(x, y)
        break