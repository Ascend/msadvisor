#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import os
import sys
import knowledges
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
from util import log

def data_process(file_pathname, extend_result):
    # 遍历该目录下的所有code文件
    if not os.path.isdir(file_pathname):
        log.ad_log(log.AD_ERROR, "The file_pathname is incorrect. Please check file_pathname path.")
        return extend_result
    log.ad_log(log.ad_info, "Start scanning file.")
    for root, dirs, files in os.walk(file_pathname):
        for file in files:
            if file.endswith('.cpp') or file.endswith('.py') or file.endswith('.h'):
                line_num = 0
                path = os.path.join(root, file)
                with open(path, encoding='UTF-8') as file:
                    for line in file.readlines():
                        line_num += 1
                        for k, r in knowledges.knowledges.items(): #遍历知识库
                            if k in line:
                                value = []
                                value.append(k)
                                value.append(r)
                                value.append(str(file.name) + ' Line:' + str(line_num))
                                extend_result.value.append(value)
    log.ad_log(log.ad_info, "Finish scanning file.")
    return extend_result