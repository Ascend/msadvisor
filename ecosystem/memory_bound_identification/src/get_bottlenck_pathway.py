#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""


# 1.1 搬运瓶颈 Memory Bound
# 1.识别数据访问瓶颈通路，使用带宽更大的数据访问通路。
def get_bottleneck_pathway(data, Topk):
    list = data
    bottleneck_pathway_list = []
    sorted_bottleneck_pathway_list = []
    threshold_value = 0.8  # 瓶颈率阈值
    # Bottleneck_Path_Rate
    l1_to_l0a_bottleneck_rate = 0
    l1_to_l0b_bottleneck_rate = 0
    ddr_to_l0a_bottleneck_rate = 0
    ddr_to_l0b_bottleneck_rate = 0
    ddr_to_l1_bottleneck_rate = 0
    ddr_to_ub_bottleneck_rate = 0
    l2_to_l0a_bottleneck_rate = 0
    l2_to_l0b_bottleneck_rate = 0
    l2_to_l1_bottleneck_rate = 0
    l2_to_ub_bottleneck_rate = 0
    ub_to_ddr_bottleneck_rate = 0
    ub_to_l1_bottleneck_rate = 0
    ub_to_l2_bottleneck_rate = 0

    for i in range(len(list)):
        Model_Name = list[i].get('Model Name')
        Op_Name = list[i].get('Op Name')
        OP_Type = list[i].get('OP Type')
        op_name = Model_Name + '/' + Op_Name + '/' + OP_Type
        l1_read_bw = float(list[i].get('l1_read_bw(GB/s)'))
        l1_write_bw = float(list[i].get('l1_write_bw(GB/s)'))
        l0a_write_bw = float(list[i].get('l0a_write_bw(GB/s)'))
        l0b_write_bw = float(list[i].get('l0b_write_bw(GB/s)'))
        main_mem_read_bw = float(list[i].get('main_mem_read_bw(GB/s)'))
        main_mem_write_bw = float(list[i].get('main_mem_write_bw(GB/s)'))
        ub_read_bw = float(list[i].get('ub_read_bw(GB/s)'))
        ub_write_bw = float(list[i].get('ub_write_bw(GB/s)'))
        l2_read_bw = float(list[i].get('l2_read_bw(GB/s)'))
        l2_write_bw = float(list[i].get('l2_write_bw(GB/s)'))
        memory_bound = float(list[i].get('memory_bound'))
        Block_Dim = float(list[i].get('Block Dim'))
        aicore_time = float(list[i].get('aicore_time(us)'))
        dict_temp2 = {}
        if memory_bound > 1:
            # MTE1
            l1_to_l0a_bottleneck_rate = min(l1_read_bw, l0a_write_bw) / 324.25 / Block_Dim
            l1_to_l0b_bottleneck_rate = min(l1_read_bw, l0b_write_bw) / 162.13 / Block_Dim
            # MTE2
            ddr_to_l0a_bottleneck_rate = min(main_mem_read_bw, l0a_write_bw) / 40.96 / Block_Dim
            ddr_to_l0b_bottleneck_rate = min(main_mem_read_bw, l0b_write_bw) / 40.96 / Block_Dim
            ddr_to_l1_bottleneck_rate = min(main_mem_read_bw, l1_write_bw) / 40.96 / Block_Dim
            ddr_to_ub_bottleneck_rate = min(main_mem_read_bw, ub_write_bw) / 40.96 / Block_Dim
            l2_to_l0a_bottleneck_rate = min(l2_read_bw, l0a_write_bw) / 81.06 / Block_Dim
            l2_to_l0b_bottleneck_rate = min(l2_read_bw, l0b_write_bw) / 81.06 / Block_Dim
            l2_to_l1_bottleneck_rate = min(l2_read_bw, l1_write_bw) / 81.06 / Block_Dim
            l2_to_ub_bottleneck_rate = min(l2_read_bw, ub_write_bw) / 81.06 / Block_Dim
            # MTE3
            ub_to_ddr_bottleneck_rate = min(ub_read_bw, main_mem_write_bw) / 40.96 / Block_Dim
            ub_to_l1_bottleneck_rate = min(ub_read_bw, l1_write_bw) / 81.06 / Block_Dim
            ub_to_l2_bottleneck_rate = min(ub_read_bw, l2_write_bw) / 40.53 / Block_Dim
            # 计算所有通路中最大的瓶颈率
            max_bottleneck_rate = max(l1_to_l0a_bottleneck_rate, l1_to_l0b_bottleneck_rate, ddr_to_l0a_bottleneck_rate, ddr_to_l0b_bottleneck_rate, ddr_to_l1_bottleneck_rate,
                                      ddr_to_ub_bottleneck_rate, l2_to_l0a_bottleneck_rate, l2_to_l0b_bottleneck_rate, l2_to_l1_bottleneck_rate, l2_to_ub_bottleneck_rate,
                                      ub_to_ddr_bottleneck_rate, ub_to_l1_bottleneck_rate, ub_to_l2_bottleneck_rate)
            # 判断是哪个通路的瓶颈率最大
            if max_bottleneck_rate == l1_to_l0a_bottleneck_rate:
                bottleneck_path = 'L1->L0A'
            if max_bottleneck_rate == l1_to_l0b_bottleneck_rate:
                bottleneck_path = 'L1->L0B'
            if max_bottleneck_rate == ddr_to_l0a_bottleneck_rate:
                bottleneck_path = 'DDR->L0A'
            if max_bottleneck_rate == ddr_to_l0b_bottleneck_rate:
                bottleneck_path = 'DDR->L0B'
            if max_bottleneck_rate == ddr_to_l1_bottleneck_rate:
                bottleneck_path = 'DDR->l1'
            if max_bottleneck_rate == ddr_to_ub_bottleneck_rate:
                bottleneck_path = 'DDR->UB'
            if max_bottleneck_rate == ub_to_ddr_bottleneck_rate:
                bottleneck_path = 'UB->DDR'
            if max_bottleneck_rate == ub_to_l1_bottleneck_rate:
                bottleneck_path = 'UB->L1'
            if max_bottleneck_rate == ub_to_l2_bottleneck_rate:
                bottleneck_path = 'UB->L2'
            if max_bottleneck_rate > threshold_value:
                dict_temp1 = {'op_name': op_name, 'bottleneck_path': bottleneck_path, 'bottleneck_rate': max_bottleneck_rate, 'aicore_time': aicore_time}
                dict_temp2.update(dict_temp1)
                bottleneck_pathway_list.append(dict_temp2.copy())
    if len(bottleneck_pathway_list) > 0:
        sorted_id = sorted(range(len(bottleneck_pathway_list)), key=lambda x: bottleneck_pathway_list[x].get('aicore_time'), reverse=True)
        for i in range(len(bottleneck_pathway_list)):
            sorted_bottleneck_pathway_list.append(bottleneck_pathway_list[sorted_id[i]])
        if len(bottleneck_pathway_list) <= Topk:
            return sorted_bottleneck_pathway_list
        else:
            return sorted_bottleneck_pathway_list[:Topk]
    else:
        return sorted_bottleneck_pathway_list
