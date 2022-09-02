#!/usr/bin/env python3.7.5
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
"""

import os
import json
import time
import model_C
import model_E
# 根据路径获取解析数据json->python



def get_data(filename, dir_path='./', second_path=''):
    file_path = os.path.join(dir_path, second_path)   # ./second_path
    file_path = os.path.join(file_path, filename)
    real_file_path = os.path.realpath(file_path)
    with open(real_file_path, 'r', encoding='UTF-8') as task_json_file:
        task_data = json.load(task_json_file)
    return task_data

# 专家系统调用的函数
def evaluate(datapath, parameter):
    environment_data = get_data('environmentConfig.json', datapath, 'knowledgeBase')   # 获取系统配置文件的数据environmentConfig.json
    version = environment_data.get('English')

    if version == 0:
        ret = model_C.Evaluate(datapath, parameter)
    else:
        ret = model_E.Evaluate(datapath, parameter)

    if not ret:
        print('The address of the knowledge base file is incorrect. Please check the file name')

    return ret

