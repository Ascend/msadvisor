#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import os
import csv
import codecs


def get_csv_path(data_path, divice_id):
    profiling_path = os.path.join(data_path, 'profiling')
    if not os.path.exists(profiling_path):
        print(f"The path of profiling does not exist, and the error path is {profiling_path}")
        os._exit(0)
    path_op_summary = []
    for file_name in os.listdir(profiling_path):
        if 'PROF' in file_name:
            a = str(divice_id)
            f1 = 'device_' + a
            f2 = 'summary'
            f3 = 'op_summary_0_1_1.csv'
            file_path = os.path.join(profiling_path, file_name, f1, f2, f3)
            path_op_summary.append(file_path)
    return path_op_summary


def get_csv_data(data_path):
    list = []
    list1 = []
    for path_op in data_path:
        d = {}
        le = 0
        with codecs.open(path_op, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f, skipinitialspace=True):
                le = le+1
                d.update(row)
                list.append(d.copy())
    # 表格数目
    num = int(len(list)/le)
    for j in range(le):
        list1.append(list[j])
    for m in range(num):
        for s in range(len(list1)):
            list1[s].update(list[s+m*le])
    return list1


def get_cce_path(data_path):
    lis = []
    profiling_path = os.path.join(data_path, 'kernel_meta')
    if not os.path.exists(profiling_path):
        print(f"The path of kernel_meta does not exist, and the error path is {profiling_path}")
        os._exit(0)
    path_op_summary = []
    for file_name in os.listdir(profiling_path):
        if '.cce' in file_name:
            file_path = os.path.join(profiling_path, file_name)
            path_op_summary.append(file_path)
            lis.append(file_name)
    return path_op_summary
