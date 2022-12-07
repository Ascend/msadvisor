# -*- coding:utf-8 -*-
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""
import json


class_type = {'op': '0', 'model': '1'}
error_code = {'success': '0', 'optimized': '1'}
summary = 'Op operations need to be optimized'

extend_type = {'list': '0', 'table': '1', 'sourcedata': '2'}
extend_data_type = {'str': '0', 'int': '1', 'double': '2'}


class ExtendResult:
    def __init__(self, title, value):
        self.type = extend_type['table']
        self.extend_title = title
        self.data_type = [
            extend_data_type['int'],  # Line
            extend_data_type['int'],  # Column
            extend_data_type['str'],  # Origin(Code)
            extend_data_type['str'],  # Advice
            extend_data_type['int'],  # AdviceNo
        ]
        self.key = ['Line', 'Column', 'Origin', 'Advice', 'AdviceNo']
        self.value = value


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
        outputstr = json.dumps(res, ensure_ascii=False)
        return outputstr

