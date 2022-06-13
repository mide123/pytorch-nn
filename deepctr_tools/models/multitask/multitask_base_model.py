import time

import numpy as np
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList

from ..basemodel import BaseModel
from layers import slice_arrays


class MultitaskBaseModel(BaseModel):

    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task_names=None, task_types=None, device='cpu',
                 gpus=None):

        super(MultitaskBaseModel, self).__init__(linear_feature_columns, dnn_feature_columns
                                                 , l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding
                                                 , init_std=init_std, seed=seed, device=device, gpus=gpus)

        self.task_names = task_names
        self.task_types = task_types

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, gpus=None):

        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]
        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []
        self.gpus = gpus

        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y)
        )
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        print("Train on {0} samples, {1} steps per epoch".format(
            len(train_tensor_data), steps_per_epoch))

        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        # 单个任务loss获取
                        pred = model(x)
                        optim.zero_grad()

                        for i in range(0, len(self.task_names)):
                            tmp_train = y_train.to(self.device).float()[:, i].to(self.device).float()
                            tmp_y = pred[i].squeeze()
                            loss = loss_func(tmp_y, tmp_train, reduction='sum')
                            total_loss = loss if i == 0 else total_loss + loss

                        optim.zero_grad()

                        reg_loss = self.get_regularization_loss()

                        total_loss = total_loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                for task_index in range(0, len(self.task_names)):
                                    metrics_task_name = f"{name}_{self.task_names[task_index]}"
                                    if metrics_task_name not in train_result:
                                        train_result[metrics_task_name] = []

                                    task_y_train = y_train[:, task_index].cpu().data.numpy()
                                    task_y_pred = pred[task_index].cpu().data.numpy().astype("float64")

                                    train_result[metrics_task_name].append(metric_fun(task_y_train, task_y_pred))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result

            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    for task_index in range(0, len(self.task_names)):
                        metrics_task_name = f"{name}_{self.task_names[task_index]}"
                        eval_str += " - " + metrics_task_name + \
                                    ": {0: .4f}".format(epoch_logs[metrics_task_name])

                if do_validation:
                    for name in self.metrics:
                        for task_index in range(0, len(self.task_names)):
                            metrics_task_name = f"val_{name}_{self.task_names[task_index]}"
                            eval_str += " - " + metrics_task_name + \
                                        ": {0: .4f}".format(epoch_logs[metrics_task_name])

                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history

    def evaluate(self, x, y, batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label1, label2) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            for task_index in range(0, len(self.task_names)):
                metrics_task_name = f"{name}_{self.task_names[task_index]}"
                eval_result[metrics_task_name] = metric_fun(y[:, task_index], pred_ans[:, task_index])

        return eval_result

    def predict(self, x, batch_size=256):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x)
                # 由于是多任务
                y_pred = torch.concat(y_pred, axis=-1)

                y_pred = y_pred.cpu().data.numpy()  # .squeeze()

                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")
