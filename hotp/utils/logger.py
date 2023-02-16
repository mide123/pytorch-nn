# -*-coding:utf-8 -*-

import json
import logging
import os

# 创建一个logs目录来存放log日志文件
from logging.handlers import RotatingFileHandler


class TslLogEntity:

    def __init__(self):
        self.total_epoch = None
        self.epoch = None
        self.loss = None
        self.lr = None
        self.Cost = None
        self.ETA = None
        self.Criterion = {}

    def get_criterion(self):
        return self.Criterion

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_total_epoch(self, Total_epoch):
        self.total_epoch = Total_epoch

    def set_loss(self, loss):
        self.loss = loss

    def set_lr(self, lr):
        self.lr = lr

    def set_cost(self, cost):
        self.Cost = cost

    def set_eta(self, eta):
        self.ETA = eta

    def set_criterion(self, key, vallue):
        self.Criterion[key] = vallue

    def __str__(self):
        return json.dumps({
            "epoch": self.epoch,
            "total_epoch": self.total_epoch,
            "loss": self.loss,
            "lr": self.lr,
            "cost": self.Cost,
            "ETA": self.ETA,
            "criterion": self.Criterion
        })


class Log(object):
    logger = logging.getLogger(__name__)

    def __init__(self, file_name, logspath="./"):
        '''
        :param file_name: 日志文件名称
        '''
        # 设置日志文件的路径
        if not os.path.exists(logspath):
            os.mkdir(logspath)
        self.file_name = file_name
        self.logfile = os.path.join(logspath, self.file_name)

        # format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        format = logging.Formatter('%(message)s')
        # 创建控制台打印输出流
        c_handler = logging.StreamHandler()
        c_handler.setFormatter(format)
        # 设置文件打印输出流 可以给2个参数，1是文件路径，2是字符集
        f_handler = logging.FileHandler(self.logfile, encoding="utf-8")
        f_handler.setFormatter(format)

        rotatingFileHandler = RotatingFileHandler(filename=f"{logspath}/bak_{file_name}", mode='a',
                                                  maxBytes=5 * 1024 * 1024, backupCount=5)

        self.logger.setLevel(logging.DEBUG)

        # 吧输出流和文件写入logger
        # AddHandler用于在运行时将事件与事件处理程序相关联
        self.logger.addHandler(f_handler)
        self.logger.addHandler(c_handler)
        self.logger.addHandler(rotatingFileHandler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)


if __name__ == '__main__':
    log = Log("test", logspath="./log/")
    log.info("first")

    log.error("first")
    log.error("first")
