#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import json


class ExtendResult:
    def __init__(self):
        self.type = '0'
        self.extend_title = ""
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
        outputstr = json.dumps(res, indent="\t", ensure_ascii=False)
        return outputstr


class_type = {'op': '0', 'model': '1'}
error_code = {'success': '0', 'optimized': '1'}
extend_type = {'list': '0', 'table': '1', 'sourcedata': '2'}
extend_data_type = {'str': '0', 'int': '1', 'double': '2'}


def result_parse(op_bottleneck_pathway_list, op_redundancy_rate_list, op_data_migration_granularity, op_bottleneck_pipeline_list,):
    # fill result
    result = Result()
    result.class_type = class_type['model']
    result.error_code = error_code['success']
    result.summary = "Operator memory operation need to be optimized"

    extend_result = ExtendResult()
    extend_result.type = extend_type['table']
    extend_result1 = ExtendResult()
    extend_result1.type = extend_type['table']
    extend_result2 = ExtendResult()
    extend_result2.type = extend_type['table']
    extend_result3 = ExtendResult()
    extend_result3.type = extend_type['table']

    # list type result 弃用此类输出格式
    if extend_result.type == '0':
        extend_result.extend_title = "Recommendations of Ops_Not_Support_Heavy_Format"
        extend_result.data_type.append(extend_data_type['str'])
        extend_result.value.append("Modify the operation to support light format")
        extend_result.value.append("Modify the operation to support heavy format")
        result.extend_result.append(extend_result)

    # table type result
    elif extend_result.type == '1':
        # 识别数据访问
        if len(op_bottleneck_pathway_list) != 0:
            extend_result.extend_title = "Op list of Memory Bottleneck Pathway and Bottleneck Rate. Suggest changing the data access path to one with higher bandwidth"
            extend_result.key.append("Op Name")
            extend_result.key.append("Bottleneck Pathway")
            extend_result.key.append("Bottleneck Rate")

            extend_result.data_type.append(extend_data_type['str'])
            extend_result.data_type.append(extend_data_type['str'])
            extend_result.data_type.append(extend_data_type['double'])

            for i in range(len(op_bottleneck_pathway_list)):
                value = []
                value.append(op_bottleneck_pathway_list[i].get('op_name'))
                value.append(op_bottleneck_pathway_list[i].get('bottleneck_path'))
                value.append(float(op_bottleneck_pathway_list[i].get('bottleneck_rate')))
                extend_result.value.append(value)
            result.extend_result.append(extend_result)

        if len(op_redundancy_rate_list) != 0:
            extend_result2.extend_title = "Op list of AiCore redundancy. Suggest reducing the amount of repeated data migration and increase FLOPS/BYTES"
            extend_result2.key.append("Op Name")
            extend_result2.key.append("AiCore Redundancy")

            extend_result2.data_type.append(extend_data_type['str'])
            extend_result2.data_type.append(extend_data_type['double'])

            for i in range(len(op_redundancy_rate_list)):
                value2 = []
                value2.append(op_redundancy_rate_list[i].get('op_name'))
                value2.append(float(op_redundancy_rate_list[i].get('aicore redundancy rate')))
                extend_result2.value.append(value2)
            result.extend_result.append(extend_result2)

        if len(op_data_migration_granularity) != 0:
            extend_result3.extend_title = "Op list of Data Migration Granularity. Please check whether data migration granularity/burst length/burst number are too small"
            extend_result3.key.append("Op Name")
            extend_result3.key.append("OUT->UB(Byte)")
            extend_result3.key.append("UB->OUT(Byte)")
            extend_result3.key.append("L1->UB(Byte)")

            extend_result3.data_type.append(extend_data_type['str'])
            extend_result3.data_type.append(extend_data_type['double'])
            extend_result3.data_type.append(extend_data_type['double'])
            extend_result3.data_type.append(extend_data_type['double'])

            for i in range(len(op_data_migration_granularity)):
                value3 = []
                value3.append(op_data_migration_granularity[i].get('op_name'))
                value3.append(float(op_data_migration_granularity[i].get('granularity_out2ub')))
                value3.append(float(op_data_migration_granularity[i].get('granularity_ub2out')))
                value3.append(float(op_data_migration_granularity[i].get('granularity_l12ub')))
                extend_result3.value.append(value3)
            result.extend_result.append(extend_result3)

        if len(op_bottleneck_pipeline_list) != 0:
            extend_result1.extend_title = "Op list of Memory Bottleneck Pipeline and Pipeline Rate. Suggest reducing unreasonable blocks inside the pipeline"
            extend_result1.key.append("Op Name")
            extend_result1.key.append("Bottleneck Pipeline")
            extend_result1.key.append("Pipeline Rate")

            extend_result1.data_type.append(extend_data_type['str'])
            extend_result1.data_type.append(extend_data_type['str'])
            extend_result1.data_type.append(extend_data_type['double'])

            for i in range(len(op_bottleneck_pipeline_list)):
                value1 = []
                value1.append(op_bottleneck_pipeline_list[i].get('op_name'))
                value1.append(op_bottleneck_pipeline_list[i].get('bottleneck_pipeline'))
                value1.append(float(op_bottleneck_pipeline_list[i].get('pipeline_rate')))
                extend_result1.value.append(value1)
            result.extend_result.append(extend_result1)
    return result.generate()
