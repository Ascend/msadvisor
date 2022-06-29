#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
"""

import msadvisor as m
import sys
import os

# msadvisor提供的打印公共接口
def ad_print(log_level, log_info):
    file_name = os.path.basename(os.path.realpath(__file__))
    file_line = str(sys._getframe(1).f_lineno)
    full_log_info = '[' + file_name + ':' +file_line + '] ' + log_info
    m.utils.print(log_level, full_log_info)

# msadvisor提供的打印并记录日志公共接口
def ad_print_and_log(log_level, log_info):
    file_name = os.path.basename(os.path.realpath(__file__))
    file_line = str(sys._getframe(1).f_lineno)
    full_log_info = '[' + file_name + ':' +file_line + '] ' + log_info
    m.utils.print_and_log(log_level, full_log_info)

# msadvisor提供的记录日志公共接口
def ad_log(log_level, log_info):
    file_name = os.path.basename(os.path.realpath(__file__))
    file_line = str(sys._getframe(1).f_lineno)
    full_log_info = '[' + file_name + ':' +file_line + '] ' + log_info
    m.utils.log(log_level, full_log_info)