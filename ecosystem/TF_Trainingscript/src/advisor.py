#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

from buildrealresult import build_simpleresult, build_locationresult
import search

class Advisor:
    def __init__(self, dtype, data):
        self.dtype = dtype
        self.data = data

    def work(self, keyword, rword, flag, path, result):
        if flag == 0 or flag == 1:
            r = search.research(self.data, keyword, rword, flag)
            build_simpleresult(r, rword, path, result)
        if flag == 2:
            r = search.concrete_research(self.data, keyword, rword)
            build_locationresult(r, rword, path, result)
        if flag == 3:
            r = search.blockresearch(self.data, keyword, rword)
            build_simpleresult(r, rword, path, result)
        if flag == '[\[]EVENT[\]]':
            r = search.plogresearch(self.data, flag, keyword, rword)
            build_simpleresult(r, rword, path, result)