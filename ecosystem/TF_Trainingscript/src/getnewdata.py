#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""
import os
from tool.python import remove_comments
import pandas as pd
import sys

path_type = {'script': 'script', 'profiling': 'profiling', 'graph': 'graph', 'plog': 'log'}

class Data:
    def __init__(self, tpath, this_path, parameter):
        self.tpath = tpath
        self.this_path = this_path
        self.parameter = parameter

    """ 获得目标目录下名称列表的数据 """
    def get_list(self, second_path):
        # 若用户指定了数据目录则读该目录下的文件
        try:
            value = self.parameter[self.tpath]
        except KeyError:
            print("key not in dict!")
        if value != '':
            file_path = value
        else:
            file_path = os.path.join(self.this_path, second_path)
        datanames = os.listdir(file_path)
        list = []
        for i in datanames:
            if self.tpath == 'script':
                if i.find('.py') == -1 and i.find('.sh') == -1:
                    continue
            if self.tpath == 'profiling':
                if i.find('.csv') == -1:
                    continue
            if self.tpath == 'graph':
                if i.find('.pbtxt') == -1 and i.find('.txt') == -1:
                    continue
            if self.tpath == 'plog':
                if i.find('.log') == -1:
                    continue
            path = os.path.join(file_path, i)
            real_file_path = os.path.realpath(path)
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
            # 若脚本代码数据不存在则无法扫描，直接退出系统
            if self.tpath == 'script':
                print("训练脚本数据不存在，请输入数据")
                sys.exit()
            print(self.tpath + '路径下无需扫描文件')
        return datalist

def processed_data(type, data_path, parameter):
    new_data = Data(type, data_path, parameter)
    data = new_data.get_datalist()
    return data

def joindata(data):
    join_data = ''
    for i in data:
        join_data += i
    return join_data
