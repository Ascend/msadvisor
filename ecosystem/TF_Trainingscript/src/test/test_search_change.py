#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import os
from model import evaluate

def get_list(second_path):
    datanames = os.listdir(second_path)
    list = []
    for i in datanames:
        path = os.path.join(second_path, i)
        real_file_path = os.path.realpath(path)
        list.append(real_file_path)
    return list

def logdelete(wordlist, path, flag):
    list = get_list(path)
    for item in list:
        f = open(item, "r", encoding='utf-8')
        data = f.read()
        for i in wordlist:
            data = data.replace(i, flag)
        f = open(item, "w", encoding='utf-8')
        f.write(data)
        f.close()

def logadd(wordlist, path, flag):
    list = get_list(path)
    for item in list:
        f = open(item, "r", encoding='utf-8')
        data = f.read()
        for i in wordlist:
            data = data.replace(flag, i)
        f = open(item, "w", encoding='utf-8')
        f.write(data)
        f.close()

def change(path, pathtest):
    f = open(path, "r", encoding='utf-8')
    data = f.read()
    ft = open(pathtest, "r", encoding='utf-8')
    datat = ft.read()
    f = open(path, "w", encoding='utf-8')
    ft = open(pathtest, "w", encoding='utf-8')
    f.truncate()
    ft.truncate()
    f.write(datat)
    ft.write(data)
    f.close()
    ft.close()

def test(word, path, flag):
    logdelete(word, path, flag)
    r = evaluate('./data/', '{"script": "", "profiling": "", "graph": "", "plog": ""}')
    print('删除关键词后结果为:\n' + r)
    logadd(word, path, flag)
    r = evaluate('./data/', '{"script": "", "profiling": "", "graph": "", "plog": ""}')
    print('还原关键词后结果为:\n' + r + '\n')

def test2(word, path, flag):
    logadd(word, path, flag)
    r = evaluate('./data/', '{"script": "", "profiling": "", "graph": "", "plog": ""}')
    print('还原关键词后结果为:\n' + r)
    logdelete(word, path, flag)
    r = evaluate('./data/', '{"script": "", "profiling": "", "graph": "", "plog": ""}')
    print('删除关键词后结果为:\n' + r + '\n')

def test3():
    r = evaluate('./data/', '{"script": "", "profiling": "", "graph": "", "plog": ""}')
    print('无优化内容时结果为:\n' + r)
    change('./data/profiling/op_summary_0_1_1.csv_test.csv', './data/profiling/summtest.csv')
    change('./data/profiling/op_statistic_0_1_1_test2.csv', './data/profiling/teststat.csv')
    r = evaluate('./data/', '{"script": "", "profiling": "", "graph": "", "plog": ""}')
    print('需优化时结果为:\n' + r + '\n')
    change('./data/profiling/op_summary_0_1_1.csv_test.csv', './data/profiling/summtest.csv')
    change('./data/profiling/op_statistic_0_1_1_test2.csv', './data/profiling/teststat.csv')