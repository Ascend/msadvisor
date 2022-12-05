#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""
import os
from tool.python import remove_comments
import pandas as pd

path_type = {'script': 'data/script', 'profiling': 'data/profiling', 'graph': 'data/graph', 'plog': 'data/log'}
 
class Data:
    def __init__(self, tpath, this_path):
        self.tpath = tpath
        self.this_path = this_path

    """ 获得目标目录下名称列表的数据 """
    def get_list(self, second_path):
        file_path = os.path.join(self.this_path, second_path)
        datanames = os.listdir(file_path)
        list = []
        for i in datanames:
            path = os.path.join(file_path, i)
            real_file_path = os.path.realpath(path)
            print(real_file_path)
            list.append(real_file_path)
        return list

    """不同扫描文件对应不同处理方法"""
    def get_datalist(self):
        list = self.get_list(path_type[self.tpath])
        datalist = []
        if list != []:
            for i in list:
                if self.tpath == 'script' or self.tpath == 'graph':
                    try:
                        # f = open(i,'rb')
                        f = open(i, encoding='utf-8')
                        data = f.read()
                        f.close()
                    except FileNotFoundError:
                        print("无法打开文件：" + i)
                    if self.tpath == 'script':
                        task_data = remove_comments(data)
                    if self.tpath == 'graph':
                        task_data = data
                if self.tpath == 'plog':
                    data = open(i, "r", encoding='utf-8')
                    task_data = data.readlines()
                if self.tpath == 'profiling':
                    task_data = pd.read_csv(i)
                datalist.append((task_data, i))
        else:
            print(self.tpath + '该路径下无需扫描文件')
        return datalist

def processed_data(type, data_path):
    new_data = Data(type, data_path)
    data = new_data.get_datalist()
    return data

def joindata(data):
    join_data = ''
    for i in data:
        join_data += i
    return join_data
