# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2022/5/24 15:51
# Author     ：heyingjie
# Description：
"""

import torch
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import History
from time_series_tool.utils.logger import Log

EarlyStopping = EarlyStopping
History = History


class EpochEndCheckpoint(ModelCheckpoint):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    Arguments:
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, out=None, **kwargs):
        super(EpochEndCheckpoint, self).__init__(**kwargs)

        if isinstance(out, Log):
            self.out_debug = out.debug
            self.out_info = out.info
            self.out_error = out.error

        else:
            self.out_info = print
            self.out_debug = print

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    self.out_debug('Can save best model only with %s available, skipping.' % self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            self.out_debug('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                           ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                                    current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            torch.save(self.model.state_dict(), filepath)
                        else:
                            torch.save(self.model, filepath)
                    else:
                        if self.verbose > 0:
                            self.out_debug('Epoch %05d: %s did not improve from %0.5f' %
                                           (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    self.out_info('Epoch %05d: saving model to %s' %
                                  (epoch + 1, filepath))
                if self.save_weights_only:
                    torch.save(self.model.state_dict(), filepath)
                else:
                    torch.save(self.model, filepath)

            # 保存每一次epoch的信息，方便下次直接提取模型
            state = {"model": self.model.state_dict(), "epoch": epoch}
            torch.save(state, filepath + "_checkpoint")
