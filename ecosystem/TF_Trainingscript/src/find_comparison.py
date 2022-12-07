#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""
import re
def isaicpu(keywords, data):
    for i in range(0, len(data) - 1):
        if (data.iloc[i]["Task Type"] == "AI_CPU"):
            for item in keywords:
                """指定的算子列表中存在AICPU算子则返回：1"""
                if data.iloc[i]["OP Type"] == item:
                    return 1

"""获取文件夹下指定的某些数据文件"""
def getprodata(type, data):
    list = []
    for i in data:
        if re.search(type, i[1]) != None:
            list.append(i[0])
    return list