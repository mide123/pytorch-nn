# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2022/5/26 15:14
# Author     ：heyingjie
# Description：
"""
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from time_series_tool.inputs import SparseFeat, DenseFeat
from time_series_tool.model.time_series import GRU
from utils import Log
from dataloader import get_mongotv_dataloader
from callbacks.callbacks import EpochEndCheckpoint
from collections import defaultdict

bash_size = 1024
device = "cuda:0"
log = Log("mongo_train", "E:/work/logs/")
window_size = 15
need_test = True
epochs = 30


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数


def get_max_min_value(df_: pd.DataFrame, gl_dict):
    df = df_[df_['cid_day_vv_t'] > 0]
    cid = df['cid_t'].values[0]
    gl_dict[cid] = (max(df['cid_day_vv_t'].values) * 0.1, min(df['cid_day_vv_t'].values * 0.1))


def predict_playback(df_: pd.DataFrame, model, feature, res_list):
    df = df_.sort_values(by=['nth_day'], ascending=True, ignore_index=True).copy(deep=True)
    # tran_data = get_mongotv_dataloader(
    #     origin_data=df,
    #     id_c='cid_t',
    #     label="cid_day_vv_t",
    #     features=feature,
    #     window_size=window_size,
    #     bash_size=bash_size,
    #     need_test=False
    # )
    # model = torch.load(model_path).eval()
    # train_model = model.train()
    # log.info(f"id_c = {df['cid_t'].values[0]}")
    # train_model.fit(tran_data, batch_size=bash_size, epochs=20)
    # model = model.eval()
    #
    train_size = len(df[df['cid_day_vv_t'] > 0])
    start_size = train_size - window_size

    while (start_size + window_size) < len(df):
        feature_values = torch.from_numpy(df[start_size: start_size + window_size][feature].values).unsqueeze(0)

        value = model.predict(feature_values)[0][0]
        df['cid_day_vv_t'][start_size + window_size] = value
        start_size += 1

    res_list.append(pd.DataFrame(df[train_size:]))


def rolling_df(df: pd.DataFrame, window=3, key='', col='', shifts=None):
    if shifts is None:
        shifts = [1, 2, 3]
    data = df.sort_values(by=[key], ascending=True)

    for shift in shifts:
        data[f'{col}_w{window}_shift{shift}_mean'] = data[col].rolling(window=window, min_periods=1).mean().shift(shift)
        data[f'{col}_w{window}_shift{shift}_max'] = data[col].rolling(window=window, min_periods=1).max().shift(shift)
        data[f'{col}_w{window}_shift{shift}_min'] = data[col].rolling(window=window, min_periods=1).min().shift(shift)
        data[f'{col}_w{window}_shift{shift}_std'] = data[col].rolling(window=window, min_periods=1).std().shift(shift)

    return pd.DataFrame(data).reset_index(drop=True)


def main():
    hash_flag = False
    same_seeds(1998)
    # data1 = pd.read_csv("../data/rank_a_data.tsv", sep='\t')
    # data2 = pd.read_csv("../data/rank_a_supp_data.tsv", sep='\t')
    data = pd.read_csv("E:/data/mangguotv/data0607/comp_2022_all_rank_b_data.tsv", sep='\t')
    # data = data1
    '''
    增加部分特征
    '''
    data['cid_t2'] = data['cid_t']
    data['nth_day_2'] = data['nth_day']
    data['data2'] = data['nth_day'].apply(lambda x: 1 if x % 2 == 0 else 0)
    data['data3'] = data['nth_day'].apply(lambda x: 1 if x % 3 == 0 else 0)
    data['data4'] = data['nth_day'].apply(lambda x: 1 if x % 4 == 0 else 0)
    data['data5'] = data['nth_day'].apply(lambda x: 1 if x % 5 == 0 else 0)
    data['data6'] = data['nth_day'].apply(lambda x: 1 if x % 6 == 0 else 0)
    data['data7'] = data['nth_day'].apply(lambda x: 1 if x % 7 == 0 else 0)
    data['data8'] = data['nth_day'].apply(lambda x: 1 if x % 8 == 0 else 0)
    data['data9'] = data['nth_day'].apply(lambda x: 1 if x % 9 == 0 else 0)

    data['kind_t1'] = data['kind_t'].apply(lambda x: int(x.split(",")[0]))
    data['kind_t2'] = data['kind_t'].apply(lambda x: -1 if len(x.split(",")) < 2 else int(x.split(",")[1]))

    data['leader_t1'] = data['leader_t'].apply(lambda x: int(x.split(",")[0]))
    data['leader_t2'] = data['leader_t'].apply(lambda x: -1 if len(x.split(",")) < 2 else int(x.split(",")[1]))

    '''
    数据归一化处理
    '''
    for feat in ['weekday', 'seriesNo', 'seriesId_t', 'channelId_t', 'kind_t1',
                 'kind_t2', 'leader_t1', 'leader_t2', 'nth_day_2', 'cid_t2']:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    max_cid_t_number = max(data['cid_t'].values)
    data['date_has_update'] = data['date_has_update'].apply(lambda x: 1 if x is True else 0)
    data['is_holiday'] = data['is_holiday'].apply(lambda x: 1 if x is True else 0)
    mms = MinMaxScaler(feature_range=(0, 1))
    data[['is_holiday', 'date_has_update', 'vv_vid_cnt', 'nth_day_2']] = \
        mms.fit_transform(data[['is_holiday', 'date_has_update', 'vv_vid_cnt', 'nth_day_2']])

    # 归一化
    cid_max_min_dict = defaultdict()
    data.groupby(['cid_t']).apply(lambda x: get_max_min_value(x, cid_max_min_dict))
    data['cid_day_vv_t'] = data[['cid_t', 'cid_day_vv_t']] \
        .apply(lambda x: 0 if x[1] == 0 else (x[1] - cid_max_min_dict[x[0]][1]) / cid_max_min_dict[x[0]][0], axis=1)
    data['cid_day_vv_t2'] = data['cid_day_vv_t'].apply(lambda x: x ** 2)
    # 对视频播放数据，在视频范围内进行归一化处理
    # window_value = 4
    # shifts = [4, 8, 12]
    # columns1 = set(data.columns)
    # data = data.groupby(['cid_t']).apply(
    #     lambda x: rolling_df(x, window=window_value, key='nth_day', col='cid_day_vv_t', shifts=shifts)).reset_index(
    #     drop=True)
    # columns2 = set(data.columns)
    # add_columns = columns2.difference(columns1)

    series_columns = [
        SparseFeat('weekday', data['weekday'].nunique(), embedding_dim=20, use_hash=hash_flag),
        SparseFeat('seriesNo', data['seriesNo'].nunique(), embedding_dim=4, use_hash=hash_flag),

        DenseFeat('cid_day_vv_t', 1),
        DenseFeat('nth_day_2', 1),
        DenseFeat('is_holiday', 1),
        DenseFeat('date_has_update', 1)
    ]
    general_features = [
        SparseFeat('cid_t2', max_cid_t_number + 1, embedding_dim=20, use_hash=hash_flag),
        SparseFeat('seriesId_t', data['seriesId_t'].nunique(), embedding_dim=20, use_hash=hash_flag),
        # SparseFeat('channelId_t', data['channelId_t'].nunique(), embedding_dim='auto', use_hash=hash_flag),
        # SparseFeat('kind_t1', data['kind_t1'].nunique(), embedding_dim='auto', use_hash=hash_flag),
        # SparseFeat('kind_t2', data['kind_t2'].nunique(), embedding_dim='auto', use_hash=hash_flag),
        # SparseFeat('leader_t1', data['leader_t'].nunique(), embedding_dim=20, use_hash=hash_flag),
        # SparseFeat('leader_t2', data['leader_t'].nunique(), embedding_dim=20, use_hash=hash_flag),
        DenseFeat('data2', 1),
        DenseFeat('data3', 1),
        DenseFeat('data4', 1),
        DenseFeat('data5', 1),
        DenseFeat('data6', 1),
        DenseFeat('data7', 1),
        DenseFeat('data8', 1),
        DenseFeat('data9', 1)
    ]

    feature_columns = series_columns + general_features

    hidden_size, num_layers, output_size = 64, 6, 1
    model = GRU(series_columns, general_features, hidden_size, num_layers, output_size,
                batch_size=bash_size,
                bidirectional=True,
                device=device,

                task='regression',
                dropout=0.7,
                log=log)
    model.compile("adam", "mae", metrics=['mae'])

    callbacks = [
        EpochEndCheckpoint(out=log, filepath="E:/work/model_file/lstm_model", save_best_only=True, verbose=1,
                           monitor='val_mae')
    ]
    features = [x.name for x in feature_columns]
    log.info("加载数据...")

    if need_test is False:
        Dtr = get_mongotv_dataloader(
            origin_data=data,
            id_c='cid_t',
            label="cid_day_vv_t",
            features=features,
            window_size=window_size,
            bash_size=bash_size,
            need_test=need_test
        )
        log.info("加载数据完成...")
        log.info("开始模型训练...")
        model.fit(Dtr, batch_size=bash_size, epochs=epochs, callbacks=callbacks)
        torch.save(model, "E:/work/model_file/lstm_model")
        # 测试环节
    else:
        Dtr, Dte = get_mongotv_dataloader(
            origin_data=data,
            id_c='cid_t',
            label="cid_day_vv_t",
            features=features,
            window_size=window_size,
            bash_size=bash_size
        )
        log.info("加载数据完成...")
        log.info("开始模型训练...")
        model.fit(Dtr, batch_size=bash_size, epochs=epochs, validation_data=Dte, callbacks=callbacks)
        # 测试环节

    # return
    log.info("输出测试数据...")
    res_list = []
    model = torch.load("E:/work/model_file/lstm_model").eval()
    data.groupby(['cid_t']).apply(lambda x: predict_playback(x, model, features, res_list))
    res_df = pd.concat(res_list).reset_index()

    res_df['cid_day_vv_t'] = res_df[['cid_t', 'cid_day_vv_t']] \
        .apply(lambda x: cid_max_min_dict[x[0]][1] + x[1] * cid_max_min_dict[x[0]][0], axis=1)

    res_df[['cid_t', 'nth_day', 'cid_day_vv_t']].to_csv("E:/work/model_file/mongh_res.csv", index=False)


def out_put_result():
    pass


if __name__ == '__main__':
    main()
