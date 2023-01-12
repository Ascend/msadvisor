#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import re


def repeat_jud(datapath):
    double_mu = ['vadd', 'vsub', 'vaddrelu', 'vsubrelu', 'vaddreluconv', 'vsubreluconv', 'vadddeqrelu',
                 'vmul', 'vmulconv', 'vdiv', 'vmax', 'vmin', 'vand', 'vor', 'scatter_vadd', 'scatter_vsub',
                 'scatter_vmul', 'scatter_vdiv', 'scatter_vmax', 'scatter_vmin', 'h_addh_sub', 'h_mul', 
                 'h_div', 'h_max', 'h_min', 'h_and', 'h_or']
    single_mu = ['vrelu', 'vexp', 'vln', 'vabs', 'vrec', 'vsqrt', 'vrsqrt', 'vnot', 'scatter_vector_mov',
                 'scatter_vrelu', 'scatter_vexp', 'scatter_vln', 'scatter_vabs', 'scatter_vrec',
                 'scatter_vsqrt', 'scatter_vrsqrt', 'h_relu', 'h_abs', 'h_not', 'h_exp', 'h_ln', 'h_rec',
                 'h_rsqrt']
    three_mu = ['vmla', 'vmadd', 'vmaddrelu', 'scatter_vmla', 'scatter_vmadd', 'scatter_vaddrelu']
    file_name = datapath
    with open(datapath, 'r', encoding='utf-8') as fd:
        lines = fd.readlines()
    fd.close()
    repeat_jud = []
    dic = {}
    for line_num, line in enumerate(lines):
        pattern = r'([a-z]\w+)\((.+)\)'
        result1 = re.findall(pattern, line)
        if result1 != []:
            result = ','.join(result1[0])
            res = result.split(',')
            if res != None:
                if res[0] in single_mu:
                    repeat = yunsuan_jud(res[4])
                    if type(repeat) == str:
                        dic['file name'] = file_name
                        dic['instruct name'] = res[0]
                        dic['line_num'] = line_num + 1
                        dic['repeat_time'] = repeat
                        dic['adv'] = 'Maybe you need to increase the number of repeats computed by Vector instructions'
                        repeat_jud.append(dic.copy())
                    elif repeat < 64:
                        dic['file name'] = file_name
                        dic['instruct name'] = res[0]
                        dic['line_num'] = line_num + 1
                        dic['repeat_time'] = str(repeat)
                        dic['adv'] = 'Increase the number of repeats computed by Vector instructions'
                        repeat_jud.append(dic.copy())
                elif (res[0] in double_mu) or (res[0] in three_mu):
                    repeat = yunsuan_jud(res[5])
                    if type(repeat) == str:
                        dic['file name'] = file_name
                        dic['instruct name'] = res[0]
                        dic['line_num'] = line_num + 1
                        dic['repeat_time'] = repeat
                        dic['adv'] = 'Maybe you need to increase the number of repeats computed by Vector instructions'
                        repeat_jud.append(dic.copy())
                    elif repeat < 64:
                        dic['file name'] = file_name
                        dic['instruct name'] = res[0]
                        dic['line_num'] = line_num + 1
                        dic['repeat_time'] = str(repeat)
                        dic['adv'] = 'Increase the number of repeats computed by Vector instructions'
                        repeat_jud.append(dic.copy())

    return repeat_jud          


def yunsuan_jud(res):
    a0 = 0
    b0 = 0
    rets = 0
    num_pattern = r'[a-zA-Z]'
    repeat1 = res
    # 判断repeat是否为变量
    if (repeat1.find('/') != -1) or (repeat1.find('%') != -1) or (repeat1.find('+') != -1) or (repeat1.find('-') != -1) or (repeat1.find('*') != -1) or (repeat1.find('[') != -1):
        rets = repeat1
        return rets
    # 判断repeat是否为单个数字（不带其他符号）
    if re.search(num_pattern, repeat1) == None:
        rets = int(repeat1)
        return rets
    for i in range(len(repeat1)):
        if repeat1[i] == ')':
            a0 = i
        if repeat1[i] == 'U':
            b0 = i
    rets = int(repeat1[a0+1:b0])
    return rets
