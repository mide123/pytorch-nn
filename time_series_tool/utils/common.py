# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2022/5/25 17:05
# Author     ：heyingjie
# Description：
"""
import os


def get_file_list(input_path):
    dir_list = []
    file_list = []

    for root, dirs, files in os.walk(input_path):
        dir_list = dirs
        file_list = files
        break

    return file_list
