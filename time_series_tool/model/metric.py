# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2022/9/1 9:54
# Author     ：heyingjie
# Description：
"""
from sklearn.metrics import *
import numpy as np
import torch


def get_accuracy_score_mul(y_true, y_pred):
    return accuracy_score(y_true, y_pred.argmax(axis=1))


def get_accuracy_score(y_true, y_pred):
    return accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0))


def get_recall_score(y_true, y_pred):
    return recall_score(y_true, np.where(y_pred > 0.5, 1, 0))


def get_recall_score_mul(indices, targets,
                         k=20):  # recall --> wether next item in session is within top K=20 recommended items or not
    """
    Calculates the recall score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
        k :
    Returns:
        recall (float): the recall score
    """
    _, indices = torch.topk(indices, k, -1)
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero()
    if len(hits) == 0:
        return 0
    n_hits = (targets == indices).nonzero()[:, :-1].size(0)
    recall = float(n_hits) / targets.size(0)
    return recall


def get_ndcg_score(targets, indices, k=20):
    """
        Calculates the MRR score for the given predictions and targets
        Args:
            targets (B): torch.LongTensor. actual target indices.
            indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
            k:
        Returns:
            ndcg (float): the ndcg score
        """
    indices = torch.from_numpy(indices)
    targets = torch.from_numpy(targets)

    _, indices = torch.topk(indices, k, -1)
    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(torch.log(ranks + 1))
    ndcg = torch.sum(rranks).data / targets.size(0)
    return ndcg
