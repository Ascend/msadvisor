#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""


# 2.流水内部发生不合理的阻塞。
def get_bottleneck_pipeline(data, Topk):
    threshold_value = 0.8  # 流水线率阈值
    list = data
    bottleneck_pipeline_list = []
    sorted_bottleneck_pipeline_list = []
    max_pipeline_time = 0
    for i in range(len(list)):
        Model_Name = list[i].get('Model Name')
        Op_Name = list[i].get('Op Name')
        OP_Type = list[i].get('OP Type')
        op_name = Model_Name + '/' + Op_Name + '/' + OP_Type
        aicore_time = float(list[i].get('aicore_time(us)'))
        vec_time = float(list[i].get('vec_time(us)'))
        mac_time = float(list[i].get('mac_time(us)'))
        scalar_time = float(list[i].get('scalar_time(us)'))
        mte1_time = float(list[i].get('mte1_time(us)'))
        mte2_time = float(list[i].get('mte2_time(us)'))
        mte3_time = float(list[i].get('mte3_time(us)'))
        memory_bound = float(list[i].get('memory_bound'))
        dict = {}
        if memory_bound > 1:
            max_pipeline_time = max(vec_time, mac_time, scalar_time, mte1_time, mte2_time, mte3_time)
            pipeline_rate = max_pipeline_time / aicore_time

            if max_pipeline_time == vec_time:
                bottleneck_pipeline = 'Vector'
            if max_pipeline_time == mac_time:
                bottleneck_pipeline = 'Cube'
            if max_pipeline_time == scalar_time:
                bottleneck_pipeline = 'Scalar'
            if max_pipeline_time == mte1_time:
                bottleneck_pipeline = 'MTE1'
            if max_pipeline_time == mte2_time:
                bottleneck_pipeline = 'MTE2'
            if max_pipeline_time == mte3_time:
                bottleneck_pipeline = 'MTE3'
            if pipeline_rate < threshold_value:
                dict = {'op_name': op_name, 'bottleneck_pipeline': bottleneck_pipeline, 'pipeline_rate': pipeline_rate, 'aicore_time': aicore_time}
                bottleneck_pipeline_list.append(dict.copy())
    if len(bottleneck_pipeline_list) > 0:
        sorted_id = sorted(range(len(bottleneck_pipeline_list)), key=lambda x: bottleneck_pipeline_list[x].get('aicore_time'), reverse=True)
        for i in range(len(bottleneck_pipeline_list)):
            sorted_bottleneck_pipeline_list.append(bottleneck_pipeline_list[sorted_id[i]])
        if len(bottleneck_pipeline_list) <= Topk:
            return sorted_bottleneck_pipeline_list
        else:
            return sorted_bottleneck_pipeline_list[:Topk]
    else:
        return sorted_bottleneck_pipeline_list
