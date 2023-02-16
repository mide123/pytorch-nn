# !/usr/bin/env python
# -*-coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from hotp.utils.logger import TslLogEntity
from .metric import *


class BaseModel(nn.Module):
    '''
    基类，主要进行模型的训练和模型的预测等工作
    '''
    def __init__(self, batchsize, device='cpu', type="regression", log=None):

        super(BaseModel, self).__init__()
        self.lr = None
        self.metrics_names = None
        self.optim = None
        self.loss_func = None
        self.metrics = None
        self.batchsize = batchsize
        self.device = device
        self.type = type
        self.log = log
        self.regularization_weight = []

    def fit(self, train, epochs=1, validation_data=None, callback=None):

        optimizer = self.optim
        loss_func = self.loss_func

        steps_per_epoch = len(train)

        train_start_time = time.time()
        log_list = []
        for epoch in range(epochs):
            model = self.train()

            log_etity = TslLogEntity()
            total_loss_epoch = 0
            train_result = {}
            for datax_y in train:
                x_list = datax_y[:-1]
                tmp_xlist = []
                for one_x in x_list:
                    tmp_xlist.append(one_x.to(self.device, dtype=torch.float))
                if self.type == "regression":
                    label = datax_y[-1].to(self.device, dtype=torch.float)
                else:
                    label = datax_y[-1].to(self.device, dtype=torch.long)
                y_pred = model(*tmp_xlist)
                loss = loss_func(y_pred, label)
                reg_loss = self.get_regularization_loss()
                loss = loss + reg_loss
                total_loss_epoch += loss.item()

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                for name, metric_fun in self.metrics.items():
                    if name not in train_result:
                        train_result[name] = []
                    train_result[name].append(metric_fun(
                        label.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

            log_etity.set_epoch(epoch)
            log_etity.set_total_epoch(epochs)
            log_etity.set_cost(int(time.time() - train_start_time))
            log_etity.set_eta(int((epochs - epoch) * (time.time() - train_start_time) / (epoch + 1)))
            log_etity.set_loss(total_loss_epoch / steps_per_epoch)
            log_etity.set_lr(self.lr)

            for name, result in train_result.items():
                log_etity.set_criterion(name, np.sum(result) / steps_per_epoch)

            if validation_data is not None:
                eval_result = self.evaluate(validation_data=validation_data)
                for name, result in eval_result.items():
                    log_etity.set_criterion("val_" + name, result)
                self.log.info(log_etity)
            log_list.append(log_etity)
            callback.on_epoch_end(model, epoch, logs=log_etity.get_criterion())
        return log_list

    def evaluate(self, validation_data):
        """

        :param validation_data: dataloader date.
        :return: Dict contains metric names and metric values.
        """

        model = self.eval()
        pred_ans = []
        y = []
        for datax_y in validation_data:
            x_list = datax_y[:-1]
            tmp_xlist = []
            for one_x in x_list:
                tmp_xlist.append(one_x.to(self.device, dtype=torch.float))
            if self.type == "regression":
                label = datax_y[-1].to(self.device, dtype=torch.float)
            else:
                label = datax_y[-1].to(self.device, dtype=torch.long)
            y_pred = model(*tmp_xlist)
            pred_ans.append(y_pred.cpu().detach().numpy())
            y.append(label.cpu())
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(np.concatenate(y).astype("float64"),
                                           np.concatenate(pred_ans).astype("float64"))

        return eval_result

    def predict(self, validation_data):
        """
        :param validation_data: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()

        pred_ans = []
        with torch.no_grad():
            y = []
            for datax_y in validation_data:
                x_list = datax_y[:-1]
                tmp_xlist = []
                for one_x in x_list:
                    tmp_xlist.append(one_x.to(self.device, dtype=torch.float))
                if self.type == "regression":
                    label = datax_y[-1].to(self.device, dtype=torch.float)
                else:
                    label = datax_y[-1].to(self.device, dtype=torch.long)
                y_pred = model(*tmp_xlist)
                pred_ans.append(y_pred.cpu().detach().numpy())
                y.append(label.cpu())

        return np.concatenate(pred_ans).astype("float64"), np.concatenate(y).astype("float64")

    def compile(self, optimizer,
                loss=None,
                metrics=None,
                lr=0.001
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        :param lr: learning rate
        """
        self.metrics_names = ["loss"]
        self.lr = lr
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=self.lr)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters(), lr=self.lr)  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters(), lr=self.lr)  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters(), lr=self.lr)
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
                if metric == "mae":
                    metrics_[metric] = mean_absolute_error
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

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        elif isinstance(weight_list, list):
            pass
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
