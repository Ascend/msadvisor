#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

from buildrealresult import get_r
import re

def research(data, keyword, rword, flag):
    searchresult = re.search(keyword, data)
    if ((searchresult == None and flag == 0) or (searchresult != None and flag == 1)):
        return get_r(rword)

def concrete_research(data, keyword, rword):
    datalines = data.splitlines()
    linelist = []
    for item in datalines:
        searchresult = re.search(keyword, item)
        if searchresult != None:
            linelist.append((datalines.index(item)+1, searchresult))
    if len(linelist) != 0:
        return (linelist, get_r(rword))
    else:
        return None

def plogresearch(data, type, keyword, rword):
    datalines = data
    for item in datalines:
        searchresult = re.search(type, item)
        if searchresult != None:
            results = re.search(keyword, item)
            if results != None:
                return None
    return get_r(rword)

def blockresearch(data, keyword, rword):
    str = re.findall(keyword[0], data)
    if str:
        for item in str:
            for i in range(1, len(keyword)):
                searchresult = re.search(keyword[i], item)
                if searchresult == None:
                    return None
            return get_r(rword)
    else:
        return None