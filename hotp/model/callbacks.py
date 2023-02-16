# !/usr/bin/env python
# -*-coding:utf-8 -*-
import torch
from hotp.utils.logger import Log
import numpy as np


class EpochEndCheckpoint:
    """
    保存最优模型
    """

    def __init__(self, out, filepath, monitor='val_mse', mode='min', save_weights_only=False):
        super(EpochEndCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_weights_only = save_weights_only

        if isinstance(out, Log):
            self.out_debug = out.debug
            self.out_info = out.info
            self.out_error = out.error
        else:
            self.out_info = print
            self.out_debug = print
            self.out_error = print

        if mode not in ['auto', 'min', 'max']:
            self.out_info('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, model, epoch, logs=None):
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch + 1)
        current = logs.get(self.monitor)
        if current is None:
            self.out_debug('Can save best model only with %s available, skipping.' % self.monitor)
        else:
            if self.monitor_op(current, self.best):
                self.out_debug('Epoch %05d: %s improved from %0.5f to %0.5f,'
                               ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                        current, filepath))
                self.best = current
                if self.save_weights_only is True:
                    torch.save(model.state_dict(), filepath)
                else:
                    torch.save(model, filepath)
            else:
                self.out_debug('Epoch %05d: %s did not improve from %0.5f' % (epoch + 1, self.monitor, self.best))
