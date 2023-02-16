#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import os
import sys
import json

def get_data(filename):
    file_path = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(file_path, filename)
    real_file_path = os.path.realpath(file_path)
    with open(real_file_path, 'r', encoding='UTF-8') as task_json_file:
        task_data = json.load(task_json_file)
    return task_data