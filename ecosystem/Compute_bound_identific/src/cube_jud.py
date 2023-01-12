#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

def cube_jud(data):
    list_cube_jud = []
    dic_com_yes = {}
    for i in range(len(data)):
        if float(data[i]['mac_ratio']) < 0.8:
            a = data[i]['Model Name']+ '/' + data[i]['Op Name'] + '/' + data[i]['OP Type']
            dic_com_yes['Op name'] = a
            dic_com_yes['aicore_time'] = float(data[i]['aicore_time(us)'])
            dic_com_yes['adv'] = 'Change calculation units, for example, replace Vector with Cube'
            list_cube_jud.append(dic_com_yes.copy())
    return list_cube_jud