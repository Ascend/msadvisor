#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import msadvisor as m
import sys
import os

ad_debug = 0
ad_info = 1
ad_warn = 2
ad_error = 3
ad_null = 4
# msadvisor提供的打印公共接口
def ad_print(log_level, log_info):
    file_name = str(os.path.basename(sys._getframe(1).f_code.co_filename))
    file_line = str(sys._getframe(1).f_lineno)   
    if log_level == ad_debug or log_level == ad_info or log_level == ad_warn or log_level == ad_error or log_level == ad_null:
        full_log_info = '[' + file_name+ ':' + file_line +'] ' + log_info
        m.utils.print(log_level, full_log_info)
    else:
        log_info_illegal = '[' + file_name+ ':' + file_line +'] ' + 'The log level is illegal, please confirm the input!'
        m.utils.log(ad_warn, log_info_illegal)
# msadvisor提供的打印并记录日志公共接口
def ad_print_and_log(log_level, log_info):
    file_name = str(os.path.basename(sys._getframe(1).f_code.co_filename))
    file_line = str(sys._getframe(1).f_lineno)
    if log_level == ad_debug or log_level == ad_info or log_level == ad_warn or log_level == ad_error or log_level == ad_null:
        full_log_info = '[' + file_name+ ':' + file_line +'] ' + log_info
        m.utils.print_and_log(log_level, full_log_info)
    else:
        log_info_illegal = '[' + file_name+ ':' + file_line +'] ' + 'The log level is illegal, please confirm the input!'
        m.utils.log(ad_warn, log_info_illegal)

# msadvisor提供的记录日志公共接口
def ad_log(log_level, log_info):
    file_name = str(os.path.basename(sys._getframe(1).f_code.co_filename))
    file_line = str(sys._getframe(1).f_lineno)
    if log_level == ad_debug or log_level == ad_info or log_level == ad_warn or log_level == ad_error or log_level == ad_null:
        full_log_info = '[' + file_name+ ':' + file_line +'] ' + log_info
        m.utils.log(log_level, full_log_info)
    else:
        log_info_illegal = '[' + file_name+ ':' + file_line +'] ' + 'The log level is illegal, please confirm the input!'
        m.utils.log(ad_warn, log_info_illegal)