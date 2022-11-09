
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
"""

import os
import sys
import json
import warnings
import function
import model_CN
import model_EN


# 专家系统调用的函数
def evaluate(datapath, parameter):
    environment_data = function.get_data(
        'environmentConfig.json',
        datapath,
        'data')   # 获取系统配置文件的数据environmentConfig.json
    language = environment_data.get('English')
    version = environment_data.get('API_version')
    datapath_pf = datapath + "/data/profiling"
    datapath_pf = function.check_profiling_data(datapath_pf)
    if version == 'V1':
        version_data = function.get_data(
            'API_version_V1.json', datapath, 'knowledgeBase')
    else:
        version_data = function.get_data(
            'API_version_V2.json', datapath, 'knowledgeBase')
    if language == 0:
        ret = model_CN.Evaluate(datapath_pf, version_data)
    else:
        ret = model_EN.Evaluate(datapath_pf, version_data)
    if not ret:
        print('The address of the knowledge base file is incorrect. Please check the file name')
    print(ret)
    return ret
evaluate("../..",1)