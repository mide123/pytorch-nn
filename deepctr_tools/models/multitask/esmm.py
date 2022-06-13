# -*- coding:utf-8 -*-
"""
Author:
    heyingjie
Reference:

"""
import torch
import torch.nn as nn

from ...inputs import combined_dnn_input
from layers import DNN, PredictionLayer
from .multitask_base_model import MultitaskBaseModel


class ESMM(MultitaskBaseModel):
    """Instantiates the Entire Space Multi-Task Model architecture.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param tower_dnn_hidden_units:  list,list of positive integer or empty list, the layer number and units in each layer of task DNN.
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN.
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task_types:  str, indicating the loss of each tasks, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss.
    :param task_names: list of str, indicating the predict target of each tasks. default value is ['ctr', 'ctcvr']
    """

    def __init__(self, dnn_feature_columns, tower_dnn_hidden_units=(256, 64), l2_reg_embedding=0.00001,
                 l2_reg_dnn=0, dnn_dropout=0, dnn_activation='relu', init_std=0.0001, seed=1024,
                 dnn_use_bn=False, task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr'),
                 device='cpu'):
        super(ESMM, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   l2_reg_embedding=l2_reg_embedding,
                                   init_std=init_std, seed=seed, task_names=task_names, task_types=task_types,
                                   device=device, gpus=device)
        if len(task_names) != 2:
            raise ValueError("the length of task_names must be equal to 2")

        for task_type in task_types:
            if task_type != 'binary':
                raise ValueError("task must be binary in ESMM, {} is illegal".format(task_type))

        self.task1 = DNN(self.compute_input_dim(dnn_feature_columns), tower_dnn_hidden_units,
                         activation=dnn_activation, dropout_rate=dnn_dropout, use_bn=dnn_use_bn, l2_reg=l2_reg_dnn)

        self.task2 = DNN(self.compute_input_dim(dnn_feature_columns), tower_dnn_hidden_units,
                         activation=dnn_activation, use_bn=dnn_use_bn, l2_reg=l2_reg_dnn)

        self.dense1 = nn.Linear(tower_dnn_hidden_units[-1], 1)
        self.dense2 = nn.Linear(tower_dnn_hidden_units[-1], 1)
        self.ctr = PredictionLayer("binary", )
        self.cvr = PredictionLayer("binary", )
        self.to(device)

    def forward(self, x):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(x, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        task1_output = self.task1(dnn_input)
        task2_output = self.task2(dnn_input)

        ctr_logit = self.dense1(task1_output)
        cvr_logit = self.dense1(task2_output)

        ctr_pred = self.ctr(ctr_logit)
        cvr_pred = self.cvr(cvr_logit)

        ctcvr_pred = torch.multiply(ctr_pred, cvr_pred)

        return ctr_pred, ctcvr_pred
