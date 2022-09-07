# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2022/5/24 11:51
# Author     ：heyingjie
# Description：
"""

try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from callbacks.callbacks import History
from ..inputs import build_input_features, create_embedding_matrix, SparseFeat, DenseFeat
from layers.core import PredictionLayer
from .metric import *


class BaseModel(nn.Module):
    def __init__(self, series_features, general_features, bashsize=32, init_std=0.0001, task='regression', device='cpu',
                 log=None):

        super(BaseModel, self).__init__()
        all_features = series_features + general_features
        self.series_features = series_features
        self.feature_index = build_input_features(all_features)

        self.regularization_weight = []
        self.embedding_dict = create_embedding_matrix(all_features, init_std, sparse=False, device=device)

        self.add_regularization_weight(self.embedding_dict.parameters(), l1=1e-5)

        self.metrics_names = None
        self.optim = None
        self.loss_func = None
        self.metrics = None
        self.bashsize = bashsize
        self.device = device
        self.print_info = log.info if log is not None else print
        self.history = History()
        self.out = PredictionLayer(task=task)
        self.to(device)

    def fit(self, train, batch_size=None, epochs=1, verbose=1, initial_epoch=0,
            validation_data=None, callbacks=None):

        optimizer = self.optim
        loss_func = self.loss_func

        steps_per_epoch = len(list(train))
        sample_num = steps_per_epoch * batch_size  # 这里还需要改

        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        for epoch in range(epochs):
            model = self.train()
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            total_loss_epoch = 0
            train_result = {}
            for (seq, label) in train:
                seq = seq.to(self.device)
                label = label.float().to(self.device)
                y_pred = model(seq)
                loss = loss_func(y_pred, label)
                reg_loss = self.get_regularization_loss()
                total_loss = loss + reg_loss
                total_loss_epoch += loss.item()

                optimizer.zero_grad()

                total_loss.backward()
                optimizer.step()

                for name, metric_fun in self.metrics.items():
                    if name not in train_result:
                        train_result[name] = []
                    train_result[name].append(metric_fun(
                        label.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if validation_data is not None:
                eval_result = self.evaluate(validation_data=validation_data)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
                self.print_info(epoch_logs)

            if verbose > 0:
                epoch_time = int(time.time() - start_time)

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if validation_data is not None:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                self.print_info(eval_str)

            callbacks.on_epoch_end(epoch, epoch_logs)
        callbacks.on_train_end()
        return self.history

    def evaluate(self, validation_data):
        """

        :param validation_data: dataloader date.
        :return: Dict contains metric names and metric values.
        """
        model = self.eval()
        pred_ans = []
        y = []
        for x_, y_ in validation_data:
            x = x_.to(self.device).float()
            y_pred = model(x).cpu().data.numpy()  # .squeeze()
            pred_ans.append(y_pred)
            y.append(y_)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(np.concatenate(y).astype("float64"),
                                           np.concatenate(pred_ans).astype("float64"))

        return eval_result

    def predict(self, x, batch_size=256):
        """
        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()

        test_loader = DataLoader(
            dataset=x, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test.to(self.device).float()
                if len(x) == 0:
                    continue
                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

    def compile(self, optimizer,
                loss=None,
                metrics=None,
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "cross_entropy":
                loss_func = F.cross_entropy
            elif loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = get_accuracy_score
                if metric == "acc_mul":
                    metrics_[metric] = get_accuracy_score_mul
                if metric == "recall_score_mul":
                    metrics_[metric] = get_recall_score_mul
                if metric == "ndcg":
                    metrics_[metric] = get_ndcg_score
                if metric == "recall_score":
                    metrics_[metric] = get_recall_score
                self.metrics_names.append(metric)
        return metrics_

    def _get_input_size(self, feature_columns):
        input_size = 0
        for feature_column in feature_columns:
            if isinstance(feature_column, SparseFeat):
                input_size += feature_column.embedding_dim
            elif isinstance(feature_column, DenseFeat):
                input_size += feature_column.dimension

        return input_size

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, :, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()).float() for
                                 feat in sparse_feature_columns]

        dense_value_list = [X[:, :, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].float() for feat
                            in
                            dense_feature_columns]

        return sparse_embedding_list, dense_value_list

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss
