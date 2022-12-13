#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import re
from getnewdata import processed_data
import buildrealresult
from advisor import Advisor
from work.script import script
from work.graph import graph
from work.log import log
from work.profiling import profiling
import sys

def evaluate(data_path, parameter):
    try:
        parameter = json.loads(parameter)
    except Exception as e:
        print(e)
    if isinstance(parameter, dict) != True:
        print("parameter应为dict格式的数据路径")
        sys.exit()
    """构造result初始值为需要优化"""
    result = buildrealresult.Result()
    try:
        result.class_type = buildrealresult.class_type['model']
        result.error_code = buildrealresult.error_code['success']
    except KeyError:
        print("key not in dict!")
    result.summary = "Training scripts need to be optimized"
    # ---------------------------------------------getdata-----------------------------------------------------
    scriptdata = processed_data('script', data_path, parameter)
    graphdata = processed_data('graph', data_path, parameter)
    plogdata = processed_data('plog', data_path, parameter)
    prodata = processed_data('profiling', data_path, parameter)
    join_scriptdata = ''
    for i in scriptdata:
        join_scriptdata += i[0]
    sadvisor = Advisor('script', join_scriptdata)
    # ------------------------------------------------work-----------------------------------------------------
    script(sadvisor, join_scriptdata, scriptdata, result)
    if graphdata != []:
        graph(graphdata, join_scriptdata, sadvisor, result)
    if prodata != []:
        profiling(prodata, scriptdata, result)
    if plogdata != []:
        log(plogdata, result)

    # list type result
    if result.extend_result == []:
        result = buildrealresult.Result()
        try:
            result.class_type = buildrealresult.class_type['model']
            result.error_code = buildrealresult.error_code['optimized']
        except KeyError:
            print("key not in dict!")
        result.summary = "Training scripts are well optimized"
    return result.generate()

if __name__ == "__main__":
    data_path = "../data/"
    ret = evaluate(data_path, '{"script": "", "profiling": "", "graph": "", "plog": ""}')
    print(ret)