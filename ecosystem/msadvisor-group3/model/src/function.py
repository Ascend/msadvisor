import re
import subprocess
import math
import os
import sys
import json
import warnings
import time
import pandas as pd
# 根据路径获取解析数据json->python


def get_data(filename, dir_path='./', second_path=''):
    file_path = os.path.join(dir_path, second_path)
    file_path = os.path.join(file_path, filename)
    real_file_path = os.path.realpath(file_path)
    with open(real_file_path, 'r') as task_json_file:
        task_data = json.load(task_json_file)
    return task_data
# 检查../../data/profiling目录中是否存在profiling文件，并检查该profiling文件是否正确配置，返回profiling文件路径
# profiling目录中只能存在一个profiling文件


def check_profiling_data(datapath):
    profiling_nums = 0
    for file in os.listdir(datapath):
        if (file[0:5] == "PROF_"):
            profiling_nums += 1
    if profiling_nums == 0:
        raise Exception(
            "profiling data do not in ../../data/profiling,or the file name is incorrect. Use the original name, such as PROF_xxxxx")
    elif profiling_nums > 1:
        raise Exception(
            "The number of profiling data is greater than 1,Please enter only one profiling data")
    datapath = datapath + '/' + os.listdir(datapath)[0]
    filename_is_correct = 1
    for file in os.listdir(datapath):
        if (file[0:7] == 'device_'):
            filename_is_correct = 0
    if filename_is_correct:
        raise ValueError(
            datapath +
            " is not a correct profiling file, correct profiling file is PROF_xxxxxxxx and it includes device_*")
    return datapath
# 检查../../data/project目录中是否存在工程项目文件


def check_project_data(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if len(file) != 0:
                return 0
    return -1


def get_statistic_profile_data(profile_path):
    for device in os.listdir(profile_path):
        path = os.path.join(profile_path, device, "summary")
        for file in os.listdir(path):
            if "acl_statistic" in file:
                acl_statistic_path = os.path.join(path, file)
    return open(acl_statistic_path)


def get_profile_data(profilepath):
    profilepath = profilepath + '/' + os.listdir(profilepath)[0]
    return profilepath
# 获取om模型同目录下的json数据


def get_project_path(path):
    path = path + '/' + os.listdir(path)[0]
    return path


def find_om_json(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if len(file.split('.')) == 2 and file.split('.')[1] == 'om':
                om_name = os.path.splitext(file)[0]
                file_list = os.listdir(root)
                for om_json_file in file_list:
                    tmp = os.path.splitext(om_json_file)
                    if tmp[0] == om_name and tmp[1] == ".json":
                        return os.path.join(root, om_json_file)
    return -1
