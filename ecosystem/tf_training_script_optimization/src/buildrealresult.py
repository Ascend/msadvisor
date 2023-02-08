#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import json

class ExtendResult:
    def __init__(self):
        self.type = '0'
        self.extend_title = [] # ""
        self.data_type = []      # table type is an array with multiple elements, list type with only one element
        self.key = []           # this field is only used for table type result
        self.value = []         # table type is a two-dimensional array, list type is a one-dimensional array

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
        outputstr = json.dumps(res)
        return outputstr

class_type = {'op': '0', 'model': '1'}
error_code = {'success': '0', 'optimized': '1'}
extend_type = {'list': '0', 'table': '1', 'sourcedata': '2'}
extend_data_type = {'str': '0', 'int': '1', 'double': '2'}

"""建议列表"""
def get_r(rword):
    task_json_file = open('recommendation.json', 'r')
    getlist = json.load(task_json_file)
    for k in getlist:
        if k == rword:
            return getlist[k]

def build_simpleresult(recommendation, rword, path, result):
    if recommendation:
        recomm_extend = ExtendResult()
        recomm_extend.type = extend_type['list']
        """"建议类别"""
        recomm_extend.identifier = rword + "-recommendations"
        """位置,文件路径"""
        recomm_extend.extend_title = "In " + path
        recomm_extend.data_type = [extend_data_type['str']]
        recomm_extend.value = recommendation
        result.extend_result.append(recomm_extend)

def build_locationresult(landr, rword, path, result):
    if landr:
        statis_identi_extend = ExtendResult()
        statis_identi_extend.type = extend_type['table']
        statis_identi_extend.identifier = rword + "-recommendations"
        statis_identi_extend.extend_title = "In " + path
        statis_identi_extend.data_types = [extend_data_type['str'], extend_data_type['str']]
        statis_identi_extend.key = ['location', 'recommendation']
        for i in landr[0]:
            statis_identi_extend.value.append(['In line: ' + str(i[0]) + ' ' + str(i[1]), landr[1]])
        result.extend_result.append(statis_identi_extend)

def build_protableresult(kandr, rword, headlist, result):
    if len(kandr) != 0:
        pvalue = []
        statis_identi_extend = ExtendResult()
        statis_identi_extend.type = extend_type['table']
        statis_identi_extend.identifier = rword + "-recommendations"
        statis_identi_extend.extend_title = get_r(rword)
        statis_identi_extend.data_types = [extend_data_type['str']]
        statis_identi_extend.key = headlist
        for i in kandr:
            for name in headlist:
                pvalue.append(i[name])
            statis_identi_extend.value.append(pvalue)
            pvalue = []
        result.extend_result.append(statis_identi_extend)