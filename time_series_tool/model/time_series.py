# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2022/5/23 19:55
# Author     ：heyingjie
# Description：
"""
import torch
import torch.nn as nn
from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from layers.interaction import CrossNet


class LSTM(BaseModel):
    def __init__(self, series_features, general_features, hidden_size, num_layers, output_size, batch_size,
                 device="cpu", task='binary', dropout=0.7,
                 log=None):
        super().__init__(series_features, general_features, batch_size, device=device, task=task, log=log)
        self.task = task
        self.device = device
        self.series_features = series_features
        self.general_features = general_features
        self.series_input_size = self._get_input_size(series_features)
        self.general_input_size = self._get_input_size(general_features)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM

        self.lstm = nn.LSTM(self.series_input_size, self.hidden_size, self.num_layers, batch_first=True,
                            dropout=dropout)

        self.cross_net = CrossNet(self.hidden_size + self.general_input_size, layer_num=2, device=device)
        self.linear = nn.Linear(self.hidden_size + self.general_input_size,
                                self.output_size)  # 这个线性层对应的是一个lstm的输出，这样最终的input_seq可以不是恒定长度

        self.add_regularization_weight(self.linear.weight, l2=1e-5)
        for weights in self.lstm.all_weights:
            self.add_regularization_weight(weights, l2=1e-5)

        self.to(device)
        self.print_info(f"input_size={self.series_input_size + self.general_input_size}")

    def forward(self, input_seq):
        bash_size = input_seq.shape[0]
        seq_len = input_seq.shape[1]
        series_sparse_embedding_list, series_dense_value_list = self.input_from_feature_columns(input_seq,
                                                                                                self.series_features,
                                                                                                self.embedding_dict)

        general_sparse_embedding_list, general_dense_value_list = self.input_from_feature_columns(input_seq,
                                                                                                  self.general_features,
                                                                                                  self.embedding_dict)

        series_combined_input = combined_dnn_input(series_sparse_embedding_list, series_dense_value_list)
        general_combined_input = combined_dnn_input(general_sparse_embedding_list, general_dense_value_list)

        h_0 = torch.zeros(self.num_directions * self.num_layers, bash_size, self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.num_directions * self.num_layers, bash_size, self.hidden_size).to(self.device)
        series_combined_input = series_combined_input.view(bash_size, seq_len, -1).float()

        series_output, _ = self.lstm(series_combined_input, (h_0, c_0))

        output = series_output.contiguous()
        cross_input = torch.cat([output, general_combined_input], dim=-1)
        cross_input = cross_input.view(bash_size * seq_len, self.hidden_size + self.general_input_size)  #

        linear_input = self.cross_net(cross_input)
        linear_input = linear_input.view(bash_size, seq_len, -1)
        linear_input = linear_input[:, -1, :]
        pred = self.linear(linear_input)
        return self.out(pred)


class GRU(BaseModel):
    def __init__(self, series_features, general_features, hidden_size, num_layers, output_size=1, bidirectional=False,
                 batch_size=21, device="cpu", task='binary', dropout=0.7,
                 log=None):
        super().__init__(series_features, general_features, batch_size, device=device, task=task, log=log)
        self.task = task
        self.device = device
        self.series_features = series_features
        self.general_features = general_features
        self.series_input_size = self._get_input_size(series_features)
        self.general_input_size = self._get_input_size(general_features)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2 if bidirectional is True else 1   # 单向LSTM

        self.gru = nn.GRU(self.series_input_size,
                          self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional,
                          dropout=dropout)

        self.cross_net = CrossNet(self.hidden_size * self.num_directions + self.general_input_size, layer_num=3, device=device)
        self.linear = nn.Linear(self.hidden_size * self.num_directions+ self.general_input_size,
                                self.output_size)  # 这个线性层对应的是一个lstm的输出，这样最终的input_seq可以不是恒定长度

        self.add_regularization_weight(self.linear.weight, l2=1e-5)
        self.add_regularization_weight(self.cross_net.kernels, l2=1e-5)
        for weights in self.gru.all_weights:
            self.add_regularization_weight(weights, l2=1e-5)

        self.to(device)
        self.print_info(f"input_size={self.series_input_size + self.general_input_size}")

    def forward(self, input_seq):
        bash_size = input_seq.shape[0]
        seq_len = input_seq.shape[1]
        series_sparse_embedding_list, series_dense_value_list = self.input_from_feature_columns(input_seq,
                                                                                                self.series_features,
                                                                                                self.embedding_dict)

        general_sparse_embedding_list, general_dense_value_list = self.input_from_feature_columns(input_seq,
                                                                                                  self.general_features,
                                                                                                  self.embedding_dict)

        series_combined_input = combined_dnn_input(series_sparse_embedding_list, series_dense_value_list)
        general_combined_input = combined_dnn_input(general_sparse_embedding_list, general_dense_value_list)

        h_0 = torch.zeros(self.num_directions * self.num_layers, bash_size, self.hidden_size).to(self.device)
        series_combined_input = series_combined_input.view(bash_size, seq_len, -1).float()

        series_output, _ = self.gru(series_combined_input, h_0)

        output = series_output.contiguous()
        cross_input = torch.cat([output, general_combined_input], dim=-1)[:, -1, :]
        cross_input = cross_input.view(bash_size, self.hidden_size * self.num_directions + self.general_input_size)  #

        linear_input = self.cross_net(cross_input)
        pred = self.linear(linear_input)
        return self.out(pred)
