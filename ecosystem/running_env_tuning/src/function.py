#!/usr/bin/env python3.7.5
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
"""
# msadvisor识别的结果类型
import re
import subprocess
import math
import os

# 函数返回状态  error:未查询到相关版本信息    overrange: 该专家系统的某调优方向不包含此推理卡
RETURN_STATUS = {'success': 0, 'optimized': 1, 'error': -1, 'overrange': -2}

# 读取文件内容
def readFile(filename, all_str):
    fopen = open(filename, 'r', encoding='ISO-8859-1')  # r 代表read
    for eachLine in fopen:
        all_str += eachLine
    return all_str


# 获取指定地址文件目录下的所有文件名
def file_name(file_dir, file_list):
    for root, dirs, files in os.walk(file_dir):
        if files:
            file_list.extend([root + path for path in files])  # 将文件相对路径放入file_list

        if dirs:
            for dir in dirs:
                file_name(root + dir + '/', file_list)
        break


# 获取指定地址文件目录下的所有文件内容
def GetFileContent(file_path_list):
    all_str = str()
    for file_path in file_path_list:
        if file_path[-1] != '/':
            file_path += '/'
        file_list = []  # 存放的是当前目录下所有文件的绝对地址
        # dirs = os.listdir(dir_path)
        file_name(file_path, file_list)
        for filepath in file_list:
            all_str = readFile(filepath, all_str)
    return all_str


# 初步获取推理卡
def getCard():
    cmd = 'lspci 2>/dev/null | grep "accelerators: Huawei Technologies Co., Ltd. Device"'
    # shell=True是为了让cmd为string（而非list，我更喜欢用string）的时候能正常执行
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 一定要decode，不然output的type会是bytes！
    output = p.stdout.decode('gb2312')
    # 获取stderr！
    error = p.stderr.decode('gb2312')
    os_version = re.findall(r'Device .{4}', output, re.M)
    return os_version[0][7:]


# 芯片为310P时，获取推理卡
def getCard310P():
    cmd = 'npu-smi info -l'
    get_npu_id = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    npu_id = get_npu_id.stdout.decode('gb2312')
    npu_id = re.findall(r'NPU ID.+: [0-9]', npu_id, re.M)
    npu_id = npu_id[-1]
    if npu_id == '0':
        cmd = 'npu-smi info -t memory -i 0'
    else:
        cmd = 'npu-smi info -t memory -i 1'
    # shell=True是为了让cmd为string（而非list，我更喜欢用string）的时候能正常执行
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 一定要decode，不然output的type会是bytes！
    output = p.stdout.decode('gb2312')
    card_capacity_all = re.findall(r'Capacity.+', output, re.M)
    sum = 0
    for i in card_capacity_all:
        card_capacity = re.findall(r'[0-9]+', i, re.M)
        sum += math.ceil(int(card_capacity[0]) / 1024)
    if sum == 24:
        return "Atlas 300I Pro"
    else:
        return "Atlas 300V Pro"


# 获取服务器操作系统版本
def getSystem_all():
    cmd = "uname -m && cat /etc/*release"
    # shell=True是为了让cmd为string（而非list，我更喜欢用string）的时候能正常执行
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 一定要decode，不然output的type会是bytes！
    output = p.stdout.decode('gb2312')
    # 获取stderr！
    error = p.stderr.decode('gb2312')
    os_version = re.findall(r'^DISTRIB_DESCRIPTION=".+"$', output, re.M)
    num = len(os_version[0])
    temp_name = os_version[0][21:num - 1]
    return temp_name


# 获取服务器操作系统版本(部分)和架构
def getSystemAndArch():
    cmd = "uname -m && cat /etc/*release"
    # shell=True是为了让cmd为string（而非list，我更喜欢用string）的时候能正常执行
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 一定要decode，不然output的type会是bytes！
    output = p.stdout.decode('gb2312')
    output = output.split("\n")
    os_Architecture = output[0]
    DISTRIB_ID = output[1][11:]
    VERSION_ID = output[10]
    VERSION_ID = VERSION_ID[12:len(VERSION_ID) - 1]
    os_version = DISTRIB_ID + " " + VERSION_ID

    return os_Architecture, os_version, DISTRIB_ID


def getInnerCoreVersion():
    cmd = "uname -srm"
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = p.stdout.decode('gb2312')
    output = output.split(" ")
    # InnerCoreversion = output[1] + "." + output[2]
    return output


# 判断该推理卡是否和当前操作系统兼容,返回是否兼容以及相关数据
def IsCompatible(data):
    temp_npu_name = getCard()
    if temp_npu_name == "d100":
        npu_name = "Atlas 300I"
    elif temp_npu_name == "d500":
        npu_name = getCard310P()
    else:
        npu_name = "others"
    os_data = getSystem_all()
    # os_data = "Windows"  # 测试
    if npu_name == "others" or npu_name == "Atlas 300I":
        return RETURN_STATUS['overrange'], npu_name, os_data
    for key in data:
        if npu_name == key:
            for os_name in data[key]:
                if os_name == os_data:
                    return RETURN_STATUS['success'], key, os_data
    return RETURN_STATUS['optimized'], npu_name, os_data


# 方向五判断内核版本是否符合，并返回判断结果以及相关信息
def IsInnerCoreAndOSCompatible(data):
    temp_npu_name = getCard()
    if temp_npu_name == "d100":
        npu_name = "Atlas 300I"
    elif temp_npu_name == "d500":
        npu_name = getCard310P()
    else:
        npu_name = "others"
    # npu_name = "Atlas 300I Pro"  # 测试
    os_Architecture, os_version, DISTRIB_ID = getSystemAndArch()  # 获取操作系统版本和架构和DISTRIB_ID(如Ubuntu)
    output = getInnerCoreVersion()  # 获取操作系统内核版本
    if DISTRIB_ID == "Ubuntu" or DISTRIB_ID == "SUSE12Sp5":
        InnerCore_version = output[1]
    else:
        InnerCore_version = output[1] + "." + output[2]
    # os_version = "Ubuntu 18.04.5"  # 测试
    if npu_name == "others" or npu_name == "Atlas 300I":
        return RETURN_STATUS['overrange'], InnerCore_version
    else:
        data = data.get(npu_name)
        for tmp in data:
            if tmp["操作系统版本"] == os_version and tmp["操作系统架构"] == os_Architecture:
                if tmp["操作系统内核默认版本"] == InnerCore_version:
                    return RETURN_STATUS['success'], InnerCore_version
                else:
                    return RETURN_STATUS['optimized'], tmp
        return RETURN_STATUS['error'], npu_name

    # 方向五判断内核版本是否符合，并返回判断结果以及相关信息


def IsInnerCoreAndOSCompatible_E(data):
    temp_npu_name = getCard()
    if temp_npu_name == "d100":
        npu_name = "Atlas 300I"
    elif temp_npu_name == "d500":
        npu_name = getCard310P()
    else:
        npu_name = "others"
    # npu_name = "Atlas 300I Pro"  # 测试
    os_Architecture, os_version, DISTRIB_ID = getSystemAndArch()  # 获取操作系统版本和架构和DISTRIB_ID(如Ubuntu)
    output = getInnerCoreVersion()  # 获取操作系统内核版本
    if DISTRIB_ID == "Ubuntu" or DISTRIB_ID == "SUSE12Sp5":
        InnerCore_version = output[1]
    else:
        InnerCore_version = output[1] + "." + output[2]
    # os_version = "Ubuntu 18.04.5"  # 测试
    if npu_name == "others" or npu_name == "Atlas 300I":
        return RETURN_STATUS['overrange'], InnerCore_version
    else:
        data = data.get(npu_name)
        for tmp in data:
            if tmp["Operating System Version"] == os_version and tmp["Operating System Architecture"] == \
                    os_Architecture:
                if tmp["Default OS kernel version"] == InnerCore_version:
                    return RETURN_STATUS['success'], InnerCore_version
                else:
                    return RETURN_STATUS['optimized'], tmp
        return RETURN_STATUS['error'], npu_name
