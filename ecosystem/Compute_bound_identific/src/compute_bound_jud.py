#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

def comp_bound(data):

    max_cube = 11140e9
    max_vector = 348.16e9
    max_cube_l0c = 648.50e9
    max_l0c_cube = 648.50e9
    max_vector_l0c = 81.06e9
    max_l0c_vector = 162.13e9
    max_vector_ub = 324.25e9
    max_ub_vector = 324.25e9
    thre_cube_l0c = max_cube/max_cube_l0c
    thre_l0c_cube = max_cube/max_l0c_cube
    thre_vector_l0c = max_vector/max_vector_l0c
    thre_l0c_vector = max_vector/max_l0c_vector
    thre_vector_ub = max_vector/max_vector_ub
    thre_ub_vector = max_vector/max_ub_vector

    list_comp_bound = []
    list_comp_well = []
    dic_com_yes = {}
    dic_com_no = {}
    for i in range(len(data)):
        if float(data[i]['mac_time(us)']) == 0:
            cal_cube = 0
        else:
            cal_cube = float(data[i]['cube_fops'])/(1e-6*float(data[i]['mac_time(us)']))   #cube实际每秒计算量
        if float(data[i]['vec_time(us)']) == 0:
            cal_vector = 0   #vector实际每秒计算量
        else:
            cal_vector = float(data[i]['vector_fops'])/(1e-6*float(data[i]['vec_time(us)']))   #vector实际每秒计算量
        if float(data[i]['l0c_write_bw_cube(GB/s)']) == 0:
            act_cube_l0c = 0
        else:
            act_cube_l0c = cal_cube/((1e9*float(data[i]['l0c_write_bw_cube(GB/s)']))/(float(data[i]['Block Dim'])))
        if float(data[i]['l0c_read_bw_cube(GB/s)']) == 0:
            act_l0c_cube = 0
        else:
            act_l0c_cube = cal_cube/((1e9*float(data[i]['l0c_read_bw_cube(GB/s)']))/(float(data[i]['Block Dim'])))
        if float(data[i]['l0c_write_bw(GB/s)']) == 0:
            act_vector_l0c = 0
        else:
            act_vector_l0c = cal_vector/((1e9*float(data[i]['l0c_write_bw(GB/s)']))/(float(data[i]['Block Dim'])))
        if float(data[i]['l0c_read_bw(GB/s)']) == 0:
            act_l0c_vector = 0
        else:
            act_l0c_vector = cal_vector/((1e9*float(data[i]['l0c_read_bw(GB/s)']))/(float(data[i]['Block Dim'])))
        if float(data[i]['ub_write_bw_vector(GB/s)']) == 0:
            act_vector_ub = 0
        else:
            act_vector_ub = cal_vector/((1e9*float(data[i]['ub_write_bw_vector(GB/s)']))/(float(data[i]['Block Dim'])))
        if float(data[i]['ub_read_bw_vector(GB/s)']) == 0:
            act_ub_vector = 0
        else:
            act_ub_vector = cal_vector/((1e9*float(data[i]['ub_read_bw_vector(GB/s)']))/(float(data[i]['Block Dim'])))
        a = data[i]['Model Name'] + '/' + data[i]['Op Name']+'/'+data[i]['OP Type']
        if (act_cube_l0c > 1.2*thre_cube_l0c) or (act_l0c_cube > 1.2*thre_l0c_cube) or (act_vector_l0c > 1.2*thre_vector_l0c) or (act_l0c_vector > 1.2*thre_l0c_vector) or (act_vector_ub > 1.2*thre_vector_ub) or (act_ub_vector > 1.2*thre_ub_vector):
            dic_com_yes['Op name'] = a
            dic_com_yes['adv1'] = 'Compute Bound'
            list_comp_bound.append(dic_com_yes.copy())
        else:
            dic_com_no['Op name'] = a
            dic_com_no['adv1'] = 'It is well optimized'
            list_comp_well.append(dic_com_no.copy())
    return list_comp_bound