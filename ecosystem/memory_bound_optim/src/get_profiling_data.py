#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import codecs
import csv
import os


# 搜索所有以‘PROF_’开头的文件夹的绝对路径
def Search_PROF_file_name(abs_path):
    list = []
    pre_data_path = abs_path + '/profiling/'
    # 切换工作目录到abspath指定目录,也就是当前工作目录
    # os.chdir(pre_data_path)
    # 列出本目录下的文件
    L = os.listdir(pre_data_path)
    for v in L:
        if os.path.isdir(pre_data_path + v) and ('PROF_' in v):  # 将本目录下的符合条件的文件夹名字输出
            PROF_data_path = pre_data_path + v  # 格式：xxx/xxx/data/profiling/PROF_xxxx
            list.append(PROF_data_path)
    return list


# 获得所有profiling文件的绝对路径，如：E:\Desktop\msadvisor-master\ecosystem\model/data/profiling/PROF_000001_20221111164735818_AENQMAPQBDAIEKCA/device_1/summary/op_summary_0_1_1.csv
def get_abs_data_path(data_path, device_id):
    data_path_list = []
    device_i = 'device_' + str(device_id)
    PROF_datapath_list = Search_PROF_file_name(data_path)

    for i in range(len(PROF_datapath_list)):
        data_path_detail = PROF_datapath_list[i] + '/' + device_i + '/summary/op_summary_0_1_1.csv'
        data_path_list.append(data_path_detail)
    return data_path_list


# 根据路径获取profiling解析数据 csv -> [{'Model Name': 'se_resnet50_fp16_bs32', ...}, {}, ...]
def get_profiling_data(data_path, device_id=0):
    data_path_list = get_abs_data_path(data_path, device_id)
    list_i = []
    for i in range(len(data_path_list)):
        locals()['list' + str(i)] = []
        with codecs.open(data_path_list[i], encoding='utf-8-sig') as f:
            for row in csv.DictReader(f, skipinitialspace=True):
                d1 = {}
                d1.update(row)
                locals()['list' + str(i)].append(d1.copy())
        list_i = locals()['list' + str(i)]
    for i in range(len(data_path_list)):
        for j in range(len(locals()['list' + str(i)])):
            list_i[j].update(locals()['list' + str(i)][j])
    return list_i

