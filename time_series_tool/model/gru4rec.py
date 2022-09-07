# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2022/6/30 16:50
# Author     ：heyingjie
# Description：
"""
from torch import nn
import time
import torch

from deepctr_tools.models.basemodel import BaseModel
from time_series_tool.inputs import combined_dnn_input


class GRU4REC(BaseModel):
    def __init__(self, series_features, hidden_size, num_layers, output_size, batch_size,
                 device="cpu", log=None, final_act='',
                 dropout_hidden=.5, dropout_input=0):
        super(GRU4REC, self).__init__(series_features, [])
        self.device = device
        self.series_features = series_features
        self.series_input_size = self._get_input_size(series_features)

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden
        self.dropout_input = dropout_input
        self.batch_size = batch_size
        self.device = device
        self.onehot_buffer = self.init_emb()
        self.h2o = nn.Linear(hidden_size, output_size)
        self.create_final_activation(final_act)

        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)

        self.to(self.device)

    def create_final_activation(self, final_act):
        if final_act == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_act == 'relu':
            self.final_activation = nn.ReLU()
        elif final_act == 'softmax':
            self.final_activation = nn.Softmax()
        elif final_act == 'softmax_logit':
            self.final_activation = nn.LogSoftmax()
        elif final_act.startswith('elu-'):
            self.final_activation = nn.ELU(alpha=float(final_act.split('-')[1]))
        elif final_act.startswith('leaky-'):
            self.final_activation = nn.LeakyReLU(negative_slope=float(final_act.split('-')[1]))

    def forward(self, input_seq, hidden):
        '''
        Args:
            input (B,): a batch of item indices from a session-parallel mini-batch.
            target (B,): torch.LongTensor of next item indices from a session-parallel mini-batch.

        Returns:
            logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
            hidden: GRU hidden state
        '''
        series_sparse_embedding_list, series_dense_value_list = self.input_from_feature_columns(input_seq,
                                                                                                self.series_features,
                                                                                                self.embedding_dict)

        series_combined_input = combined_dnn_input(series_sparse_embedding_list, series_dense_value_list)
        embedded = series_combined_input.unsqueeze(0)

        output, hidden = self.gru(embedded, hidden)  # (num_layer, B, H)
        output = output.view(-1, output.size(-1))  # (B,H)
        logit = self.final_activation(self.h2o(output))

        return logit, hidden

    def init_emb(self):
        '''
        Initialize the one_hot embedding buffer, which will be used for producing the one-hot embeddings efficiently
        '''
        onehot_buffer = torch.FloatTensor(self.batch_size, self.output_size)
        onehot_buffer = onehot_buffer.to(self.device)
        return onehot_buffer

    def onehot_encode(self, input):
        """
        Returns a one-hot vector corresponding to the input
        Args:
            input (B,): torch.LongTensor of item indices
            buffer (B,output_size): buffer that stores the one-hot vector
        Returns:
            one_hot (B,C): torch.FloatTensor of one-hot vectors
        """
        self.onehot_buffer.zero_()
        index = input.view(-1, 1)
        one_hot = self.onehot_buffer.scatter_(1, index, 1)
        return one_hot

    def embedding_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), 1).fill_(1 - self.dropout_input)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.dropout_input)
        mask = mask.to(self.device)
        input = input * mask
        return input

    def init_hidden(self, predict=False, device="cpu"):
        '''
        Initialize the hidden state of the GRU
        '''
        if predict is True:
            h0 = torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
        else:
            try:
                h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
            except:
                self.device = 'cpu'
                h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        return h0