#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import os
import json
from collections import defaultdict

import config


class ScanFile:  # 传入参数主路径，递归搜索所有py和sh文件，以路径形式加入list返回
    def __init__(self):
        pass

    def listdir(self, path, list_name):  # 传入存储的list
        lst = os.listdir(path)
        for file in lst:
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                self.listdir(file_path, list_name)
            if file.endswith('.py') or file.endswith('.sh'):
                list_name.append(file_path)

        return list_name

    def run(self, path):
        filepath = []
        result = self.listdir(path, filepath)
        return result


class ExtendResult:
    def __init__(self):
        self.type = '0'
        self.extend_title = ""
        self.data_type = []     # table type is an array with multiple elements, list type with only one element
        self.key = []           # this field is only used for table type result
        self.value = []         # table type is a two-dimensional array, list type is a one-dimensional array

    def generate(self):
        res = {
            "type": self.type,
            "dataType": self.data_type,
            "extendTitle": self.extend_title,
            "key": self.key,
            "value": self.value
        }
        return res


class Result:
    def __init__(self):
        self.class_type = '0'
        self.error_code = '0'
        self.summary = ""
        self.extend_result = []

    def generate(self):
        return json.dumps(
            {
                "classType": self.class_type,
                "errorCode": self.error_code,
                "summary": self.summary,
                "extendResult": [
                    {
                        "type": item.type,
                        "extendTitle": item.extend_title,
                        "dataType": item.data_type,
                        "key": item.key,
                        "value": item.value
                    }
                    for item in self.extend_result
                ]
            },
            indent='\t',
            ensure_ascii=False
        )


class Advisor:
    file_processors = defaultdict(list)
    final_processors = []

    def __init__(self, data_path) -> None:
        self.vars = {}
        self.files = []
        self.result_general = []
        self.result_lrbs = []
        self.result_taskset = []
        self.result = Result()
        scanfile = ScanFile()
        self.files = scanfile.run(data_path)

    @classmethod
    def register_file_processor(cls, filetype):
        def wrapper(func):
            cls.file_processors[filetype].append(func)
            return func
        return wrapper

    @classmethod
    def register_final_processor(cls, func):
        cls.final_processors.append(func)
        return func

    def add_advice(self, advice):
        pass

    def run_file(self):
        for file in self.files:
            with open(file) as f:
                content = f.readlines()
                filetype = os.path.splitext(file)[1]
                if filetype == '':
                    pass  # TODO: 一种特殊情况，没有后缀
                if filetype not in self.file_processors:
                    continue
                for p in self.file_processors[filetype]:
                    p.__func__(self, file, content)

    def run_final(self):
        for p in self.final_processors:
            p.__func__(self)

    def run(self):
        self.run_file()
        self.run_final()

    def output(self):
        if not (self.result_general or self.result_lrbs or self.result_taskset):
            self.result.class_type = config.CLASS_TYPE['model']
            self.result.error_code = config.ERROR_CODE['optimized']
            self.result.summary = config.SUMMARY['optimized']
            return self.result.generate()

        self.result.class_type = config.CLASS_TYPE['model']
        self.result.error_code = config.ERROR_CODE['success']  # 成功定位到问题
        self.result.summary = config.SUMMARY['optimizable']

        general_extend = ExtendResult()
        general_extend.type = config.EXTEND_TYPE['table']
        general_extend.extend_title = "Recommendations for various operators"
        general_extend.key = ['path', 'line', 'anchor', 'advice']
        general_extend.data_type = [
            config.EXTEND_DATA_TYPE["str"] for _ in general_extend.key
        ]
        for value in self.result_general:
            general_extend.value.append(value)
        self.result.extend_result.append(general_extend)

        lrbs_extend = ExtendResult()
        lrbs_extend.type = config.EXTEND_TYPE['table']
        lrbs_extend.extend_title = "Recommendations for learning rate or batch size"
        lrbs_extend.key = ['lr_path', 'lr_line', 'lr_anchor', 'bs_path', 'bs_line', 'bs_anchor', 'advice']
        lrbs_extend.data_type = [
            config.EXTEND_DATA_TYPE["str"] for _ in lrbs_extend.key
        ]
        for value in self.result_lrbs:
            lrbs_extend.data_type.append(config.EXTEND_DATA_TYPE["str"])
            lrbs_extend.value.append(value)
        self.result.extend_result.append(lrbs_extend)

        taskset_extend = ExtendResult()
        taskset_extend.type = config.EXTEND_TYPE['table']
        taskset_extend.extend_title = "Recommendations for taskset"
        taskset_extend.key = ['path', 'advice']
        taskset_extend.data_type = [
            config.EXTEND_DATA_TYPE["str"] for _ in taskset_extend.key
        ]
        for value in self.result_taskset:
            taskset_extend.value.append(value)
        self.result.extend_result.append(taskset_extend)

        return self.result.generate()
