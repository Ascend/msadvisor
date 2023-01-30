#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import os
import struct
import ge_ir_pb2

MAGIC_NUM = b'IMOD'
MODEL_HEAD_SIZE = 256
PARTITION_NUM_SIZE = 4
PARTITION_INFO_SIZE = 12


# 解析om模型
def parse_om_model(string):
    if len(string) < len(MAGIC_NUM) or (string[:len(MAGIC_NUM)] != MAGIC_NUM):
        return None
    string = string[MODEL_HEAD_SIZE:]

    if len(string) < PARTITION_NUM_SIZE:
        return None
    partition_num = struct.unpack('I', string[:PARTITION_NUM_SIZE])[0]
    if partition_num == 0:
        return None
    string = string[PARTITION_NUM_SIZE:]
    partitions = string[partition_num * PARTITION_INFO_SIZE:]

    for idx in range(partition_num):
        partition_info = struct.unpack('iII', string[idx * PARTITION_INFO_SIZE:(idx + 1) * PARTITION_INFO_SIZE])
        ptype, offset, size = partition_info
        if ptype == 0:
            model = ge_ir_pb2.ModelDef()
            model.ParseFromString(partitions[offset:offset + size])
            return model

    return None


# 将解析过的om模型存入txt文档
def om_model_parse_to_txt(data_path):
    om_data_path = data_path+'/project/'
    # 列出本目录下的文件
    L = os.listdir(om_data_path)
    for v in L:
        if os.path.isfile(om_data_path + v) and ('.om' in v):  # 将本目录下的符合条件的文件夹名字输出
            om_file_path = om_data_path + v  # 格式：xxx/src/../data/projrct/xxx.om
    if os.path.exists(om_file_path):
        with open(om_file_path, 'rb') as fd:
            string = fd.read()
        model = parse_om_model(string)
        with open('om_model.txt', 'w') as f:
            f.write(str(model))
    else:
        print('Please input om file!')
