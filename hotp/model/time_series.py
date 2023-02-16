# !/usr/bin/env python
# -*-coding:utf-8 -*-
import torch
import torch.nn as nn
from .basemodel import BaseModel
import torch.nn.functional as F


class LSTM(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, device="cpu", window_size=11,
                 bidirectional=False,
                 log=None):
        super().__init__(batch_size, device=device, log=log)
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        if bidirectional is True:
            self.num_directions = 2  # 单向LSTM
        else:
            self.num_directions = 1  # 单向LSTM
        self.window_size = window_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=bidirectional,
                            batch_first=True)
        self.linear = nn.Linear(self.hidden_size * self.num_directions,
                                self.output_size)  # 这个线性层对应的是一个lstm的输出，这样最终的input_seq可以不是恒定长度
        # self.dropout = nn.Dropout(0.3)
        self.add_regularization_weight(self.lstm.all_weights[0], l1=0.00001, l2=0.0)  # 正则化
        self.add_regularization_weight(self.linear.weight, l1=0.00001, l2=0.0)

        self.bn = nn.BatchNorm2d(num_features=1)  # bn层

    def forward(self, input_seq):
        input_seq = self.bn(input_seq.unsqueeze(1)).squeeze(1)  # bn层
        bash_size = input_seq.shape[0]
        seq_len = input_seq.shape[1]
        device = f"cuda:{input_seq.get_device()}" if input_seq.get_device() >= 0 else "cpu"

        h_0 = torch.zeros(self.num_directions * self.num_layers, bash_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_directions * self.num_layers, bash_size, self.hidden_size, device=device)

        output, _ = self.lstm(input_seq, (h_0, c_0))  #
        output = output.contiguous().view(bash_size * seq_len, self.hidden_size * self.num_directions)  #
        pred = self.linear(output)
        pred = pred.view(bash_size, seq_len, -1)
        pred = pred[:, -1, :]
        return pred


class GRU(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, device="cpu", window_size=11,
                 bidirectional=False,
                 log=None):
        super().__init__(batch_size, device=device, log=log)
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        if bidirectional is True:
            self.num_directions = 2  # 单向LSTM
        else:
            self.num_directions = 1  # 单向LSTM
        self.window_size = window_size
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, bidirectional=bidirectional,
                          batch_first=True)
        self.linear = nn.Linear(self.hidden_size * self.num_directions,
                                self.output_size)  # 这个线性层对应的是一个lstm的输出，这样最终的input_seq可以不是恒定长度
        # self.dropout = nn.Dropout(0.3)
        self.add_regularization_weight(self.gru.all_weights[0], l1=0.00001, l2=0.0)  # 正则化
        self.add_regularization_weight(self.linear.weight, l1=0.00001, l2=0.0)

        self.bn = nn.BatchNorm2d(num_features=1)  # bn层

    def forward(self, input_seq):
        input_seq = self.bn(input_seq.unsqueeze(1)).squeeze(1)  # bn层
        bash_size = input_seq.shape[0]
        seq_len = input_seq.shape[1]
        device = f"cuda:{input_seq.get_device()}" if input_seq.get_device() >= 0 else "cpu"

        h_0 = torch.zeros(self.num_directions * self.num_layers, bash_size, self.hidden_size, device=device)

        output, _ = self.gru(input_seq, h_0)  #
        output = output.contiguous().view(bash_size * seq_len, self.hidden_size * self.num_directions)  #
        pred = self.linear(output)
        pred = pred.view(bash_size, seq_len, -1)
        pred = pred[:, -1, :]
        return pred


class InteractingLayer(nn.Module):
    """A Layer used in AutoInt that model the correlations between different feature fields by multi-head self-attention mechanism.
      Input shape
            - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
            - 3D tensor with shape:``(batch_size,field_size,embedding_size)``.
      Arguments
            - **in_features** : Positive integer, dimensionality of input features.
            - **head_num**: int.The head number in multi-head self-attention network.
            - **use_res**: bool.Whether or not use standard residual connections before output.
            - **seed**: A Python integer to use as random seed.
      References
            - [Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """

    def __init__(self, embedding_size, head_num=2, use_res=True, scaling=False, seed=1024, device='cpu'):
        super(InteractingLayer, self).__init__()
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        if embedding_size % head_num != 0:
            raise ValueError('embedding_size is not an integer multiple of head_num!')
        self.att_embedding_size = embedding_size // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.seed = seed

        self.W_Query = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))

        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

        self.to(device)

    def forward(self, inputs):

        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))

        # None F D
        querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        # head_num None F D/head_num
        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)  # head_num None F F
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = F.softmax(inner_product, dim=-1)  # head_num None F F
        result = torch.matmul(self.normalized_att_scores, values)  # head_num None F D/head_num

        result = torch.cat(torch.split(result, 1, ), dim=-1)
        result = torch.squeeze(result, dim=0)  # None F D
        if self.use_res:
            result += torch.tensordot(inputs, self.W_Res, dims=([-1], [0]))
        result = F.relu(result)

        return result


class AttentionLSTM(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, device="cpu", window_size=11,
                 bidirectional=False,
                 log=None):
        super().__init__(batch_size, device=device, log=log)
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        if bidirectional is True:
            self.num_directions = 2  # 单向LSTM
        else:
            self.num_directions = 1  # 单向LSTM
        self.window_size = window_size
        self.attention = InteractingLayer(self.input_size, 1)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=bidirectional,
                            batch_first=True)
        self.linear = nn.Linear(self.hidden_size * self.num_directions,
                                self.output_size)  # 这个线性层对应的是一个lstm的输出，这样最终的input_seq可以不是恒定长度
        # self.dropout = nn.Dropout(0.3)
        self.add_regularization_weight(self.lstm.all_weights[0], l1=0.00001, l2=0.0)  # 正则化
        self.add_regularization_weight(self.linear.weight, l1=0.00001, l2=0.0)

        self.bn = nn.BatchNorm2d(num_features=1)  # bn层

    def forward(self, input_seq):
        input_seq = self.bn(input_seq.unsqueeze(1)).squeeze(1)  # bn层
        bash_size = input_seq.shape[0]
        seq_len = input_seq.shape[1]
        device = f"cuda:{input_seq.get_device()}" if input_seq.get_device() >= 0 else "cpu"

        h_0 = torch.zeros(self.num_directions * self.num_layers, bash_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_directions * self.num_layers, bash_size, self.hidden_size, device=device)
        input_seq = self.attention(input_seq)
        output, _ = self.lstm(input_seq, (h_0, c_0))  #
        output = output.contiguous().view(bash_size * seq_len, self.hidden_size * self.num_directions)  #
        pred = self.linear(output)
        pred = pred.view(bash_size, seq_len, -1)
        pred = pred[:, -1, :]
        return pred


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="cpu"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.batch_size = batch_size
        self.device = device
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))

        return h, c


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, step_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size
        self.step_size = step_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq, h, c):
        batch_size = input_seq.shape[0]
        input_seq = input_seq.view(batch_size, 1, self.input_size)
        for i in range(self.step_size+3):
            input_seq, (h, c) = self.lstm(input_seq, (h, c))
        pred = self.linear(input_seq)
        pred = pred[:, -1, :]

        return pred


class Seq2Seq(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, device="cpu", step_size=0, log=None):
        super().__init__(batch_size, device=device, log=log)
        self.device= device
        self.output_size = output_size
        self.Encoder = Encoder(input_size, hidden_size, num_layers, batch_size, device=device)
        self.Decoder = Decoder(input_size, input_size, num_layers, output_size, batch_size, step_size)
        self.step_size = step_size

    def forward(self, input_seq):
        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        h, c = self.Encoder(input_seq)
        output = self.Decoder(input_seq[:, -1, :], h, c)

        return output

class CnnGru(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, device="cpu", bidirectional=False,
                 log=None):
        super().__init__(batch_size, device=device, log=log)
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        if bidirectional is True:
            self.num_directions = 2  # 单向LSTM
        else:
            self.num_directions = 1  # 单向LSTM
        self.cnn_list = nn.ModuleList(nn.Conv2d(1, 1, kernel_size=(5, 1), padding=0) for _ in range(3))
        self.pool_layer = nn.MaxPool2d((4, 1), padding=0)
        self.lstm = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(self.hidden_size * self.num_directions,
                                self.output_size)  # 这个线性层对应的是一个lstm的输出，这样最终的input_seq可以不是恒定长度

    def get_con_values(self, input, model_list: nn.ModuleList):
        sensor_input = input.unsqueeze(1)
        for conv_layer in model_list:
            sensor_input = conv_layer(sensor_input)
            sensor_input = self.pool_layer(sensor_input)
            sensor_input = torch.nn.Tanh()(sensor_input)
        return sensor_input.squeeze(1)

    def forward(self, input_seq):
        bash_size = input_seq.shape[0]
        device = f"cuda:{input_seq.get_device()}" if input_seq.get_device() >= 0 else "cpu"

        h_0 = torch.zeros(self.num_directions * self.num_layers, bash_size, self.hidden_size, device=device)

        input_seq = self.get_con_values(input_seq, self.cnn_list)
        seq_len = input_seq.shape[1]
        output, _ = self.lstm(input_seq, h_0)  #
        output = output.contiguous().view(bash_size * seq_len, self.hidden_size * self.num_directions)  #
        pred = self.linear(output)
        pred = pred.view(bash_size, seq_len, -1)
        pred = pred[:, -1, :]
        return pred