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
import model_E_gai
import model_C_gai
# 根据路径获取解析数据json->python



def get_data(filename, dir_path='./', second_path=''):
    file_path = os.path.join(dir_path, second_path)   # ./second_path
    file_path = os.path.join(file_path, filename)
    real_file_path = os.path.realpath(file_path)
    with open(real_file_path, 'r', encoding='UTF-8') as task_json_file:
        task_data = json.load(task_json_file)
    return task_data

# 专家系统调用的函数
def Evaluate(datapath):
    environment_data = get_data('environmentConfig.json', datapath, 'knowledgeBase')   # 获取系统配置文件的数据environmentConfig.json
    version = environment_data.get('English')

    if version == 0:
        ret = model_C_gai.Evaluate(datapath)
    else:
        ret = model_E_gai.Evaluate(datapath)

    if not ret:
        print('The address of the knowledge base file is incorrect. Please check the file name')

    return ret

# 主函数接口，在本地调试试使用
if __name__ == '__main__':
    environment_data = get_data('environmentConfig.json', './', 'knowledgeBase')   # 获取系统配置文件的数据environmentConfig.json
    version = environment_data.get('English')
    time_start = time.time()
    datapath = './'

    if version == 0:
        ret = model_C.Evaluate(datapath)
    else:
        ret = model_E.Evaluate(datapath)
    if ret==False:
            print('The address of the knowledge base file is incorrect. Please check the file name')
    else:
        print(ret.encode('utf-8').decode('unicode_escape'))   # 将unicode编码妆化为中文
    time_end = time.time()
    # print('elapsed time', time_end-time_start)  # 此处单位为秒
