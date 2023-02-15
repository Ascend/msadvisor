#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import os
import knowledges


def data_process(file_pathname, extend_result):
    # 遍历该目录下的所有code文件
    if not os.path.isdir(file_pathname):
        return extend_result

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

    return extend_result