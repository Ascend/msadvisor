#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import json
import os
import config
import method
from util import *



class ExtendResult:
    def __init__(self):
        self.type = '0'
        self.extend_title = ""
        self.data_type = []      # table type is an array with multiple elements, list type with only one element
        self.key = []           # this field is only used for table type result
        self.value = []         # table type is a two-dimensional array, list type is a one-dimensional array

    def generate(self):
        res = {"type": self.type, "dataType": self.data_type, "extendTitle": self.extend_title,
                "key": self.key, "value": self.value}
        return res


class Result:
    def __init__(self):
        self.class_type = '0'
        self.error_code = '0'
        self.summary = ""
        self.extend_result = []

    def generate(self):
        extend_data = []
        for item in self.extend_result:
            data = {"type": item.type, "extendTitle": item.extend_title,
                    "dataType": item.data_type, "key": item.key, "value": item.value}
            extend_data.append(data)
        res = {"classType": self.class_type, "errorCode": self.error_code,
               "summary": self.summary, "extendResult": extend_data}
        outputstr = json.dumps(res,indent='\t',ensure_ascii=False)#添加了格式化和中文乱码问题

        return outputstr


class advisor:
    def __init__(self, data_path) -> None:
        self.vars = {}
        self.files = []
        self.file_processors = {}
        self.final_processors = []
        self.result_lst = []
        self.result = Result()
        scanfile = method.ScanFile()
        self.files = scanfile.run(data_path)


    def register_file_processor(self, filetype, func):
        if filetype not in self.file_processors:
            self.file_processors[filetype] = []
        self.file_processors[filetype].append(func)


    def register_final_processor(self, func):
        self.final_processors.append(func)


    def add_advice(self, advice):
        pass


    def run_file(self):
        for file in self.files:
            with open(file) as f:
                content = f.readlines()
                filetype = os.path.splitext(file)[1]
                if filetype == '':
                    pass # TODO: 一种特殊情况，没有后缀
                if filetype not in self.file_processors:
                    continue
                for p in self.file_processors[filetype]:
                    p(self, file, content)


    def run_final(self):
        for p in self.final_processors:
            p(self)


    def run(self):
        self.run_file()
        self.run_final()


    def output(self):
        if len(self.result_lst) == 0:
            self.result.class_type = config.CLASS_TYPE['model']
            self.result.error_code = config.ERROR_CODE['optimized']
            self.result.summary = config.SUMMARY['optimized']
            return self.result.generate()

        self.result.class_type = config.CLASS_TYPE['model']
        self.result.error_code = config.ERROR_CODE['success']#成功定位到问题
        self.result.summary = config.SUMMARY['optimizable']

        # 创建ExtendResult
        model_identi_extend = ExtendResult()
        model_identi_extend.type = config.EXTEND_TYPE['list']
        model_identi_extend.extend_title = "Recommendations"

        for value in self.result_lst:
            model_identi_extend.data_type.append(config.EXTEND_DATA_TYPE["str"])
            model_identi_extend.value.append(value)

        self.result.extend_result.append(model_identi_extend)
        return self.result.generate()

def evaluate(data_path, parameter):
    """
    interface function called by msadvisor
    Args:
        data_path: string data_path
        parameter: string parameter
    Returns:
        json string of result info
        result must by ad_result
    """

    # do evaluate work by file data
    if not os.path.exists(data_path):
        print("file or dir:", data_path, "no exist")
        raise FileNotFoundError

    ad = advisor(data_path)

    method.task1(ad)
    method.task2(ad)
    method.task3(ad)
    method.task4(ad)
    method.task5(ad)

    ad.run()

    return ad.output()


if __name__ == "__main__":

    project_path = "../data/project/"

    ret = evaluate(project_path, "none")
    print("----------result:----------")
    print(ret)
