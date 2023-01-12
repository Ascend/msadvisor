#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

def bank_jud(data):
    thre_bank_conflic = 0.05
    list_bank_jud = []
    dic_com_yes = {}    
    for i in range(len(data)):
        if (float(data[i]['vec_bankgroup_cflt_ratio']) > thre_bank_conflic)or(float(data[i]['vec_bank_cflt_ratio'])>thre_bank_conflic):
            a = data[i]['Model Name']+'/'+ data[i]['Op Name']+'/'+data[i]['OP Type']
            dic_com_yes['Op name'] = a  
            dic_com_yes['aicore_time'] = float(data[i]['aicore_time(us)'])
            dic_com_yes['adv_1'] = 'Latency compute bound'
            dic_com_yes['adv'] = 'Check bank conflict'
            list_bank_jud.append(dic_com_yes.copy())
    return list_bank_jud