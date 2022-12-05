#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import re
from getnewdata import processed_data, joindata
import buildrealresult
from advisor import Advisor
import find_comparison

def evaluate(data_path, parameter):
    """构造result初始值为需要优化"""
    result = buildrealresult.Result()
    result.class_type = buildrealresult.class_type['model']
    result.error_code = buildrealresult.error_code['success']
    result.summary = "Training scripts need to be optimized"

    # ---------------------------------------------script------------------------------------------------------
    scriptdata = processed_data('script', data_path)
    join_scriptdata = ''
    for i in scriptdata:
        join_scriptdata += i[0]
    sadvisor = Advisor('script', join_scriptdata)

    # 训练脚本中不存在tf.data.Dataset字段
    sadvisor.work('tf.data.Dataset', 'The Dataset interface', 0, "TF Training script", result)
    # 训练脚本中不包含NPURunConfig，不存在enable_data_pre_proc字段
    if re.search('NPURunConfig', join_scriptdata) == None:
        sadvisor.work('enable_data_pre_proc', 'Gennext sinking', 0, "TF Training script", result)
    # 不存在allow_mix_precision字段
    sadvisor.work('allow_mix_precision', "Mixing precision", 0, "TF Training script", result)
    # set_split_strategy_by_idx或者set_split_strategy_by_size没有设置，hcom_parallel未设置为True
    sadvisor.work('set_split_strategy_by', "Allreduce", 0, "TF Training script", result)
    sadvisor.work('hcom_parallel\s*=\s*True', "Allreduce", 0, "TF Training script", result)

    if scriptdata != []:
        for i in scriptdata:
            advisor = Advisor('script', i[0])

            # 扫描训练脚本中是否使用了溢出检测API
            advisor.work('(tf.)(debugging.|math.|compat.v1.debugging.|compat.v1.|compat.v1.math.)?(is_finite|is_nan|is_inf)',
                          'loss scale', 2, i[1], result)
            # # 没有npu_ops.dropout但训练脚本中存在dropout字段输出建议，有npu_ops.dropout则不提建议
            # if re.search('npu_ops.dropout', i[0]) == None:
            #     advisor.work('(?<!npu_ops.)dropout', "dropout interface", 2, i[1], result)
            # 使用了原始小算子自定义gelu函数
            advisor.work('def gelu[(]', "gelu interface", 2, i[1], result)
            # 存在mix_compile_mode字段
            advisor.work('mix_compile_mode', "mixed calculation", 2, i[1], result)
            # 存在tf.boolean_mask字段
            advisor.work('tf.boolean_mask', "interface1", 2, i[1], result)
            # LSTM接口
            advisor.work('(tf.)(nn.rnn_cell.BasicLSTMCell|contrib.rnn.LayerNormBasicLSTMCell|contrib.rnn.BasicLSTMCell)',
                          'LSTM', 2, i[1], result)
            # keras框架 tensorflow.keras，xxx.compile(xxx,loss="categorical_crossentropy", metrics=["accuracy"])
            if re.search('tensorflow.keras', i[0]) != None:
                advisor.work(['\.compile[(]([\s\S]*),\n\s*[)]','metrics\s*=\s*[\[]"accuracy"[\]]',
                              'loss\s*=\s*"categorical_crossentropy"'], 'interface2', 3, i[1], result)
            # Estimator模式=0 sess.run模式=1 Keras模式=2
            if re.search('tensorflow.keras', i[0]) != None:
                mode = re.search('\.compile[(]', i[0])
                if (mode != None):
                    findlist = list(set(re.compile('(npu_distributed_optimizer_wrapper)|(NPUBroadcastGlobalVariablesCallback)'
                                                   ).findall(i[0])))
                    if len(findlist) == 0:
                        advisor.work('(npu_distributed_optimizer_wrapper)|(NPUBroadcastGlobalVariablesCallback)',
                             'Keras data parallel', 0, "TF Training script", result)
                    else:
                        if len(findlist) != 2:
                            advisor.work('(npu_distributed_optimizer_wrapper)|(NPUBroadcastGlobalVariablesCallback)',
                                 'Keras data parallel', 2, i[1], result)
            mode = re.search('(.*).estimator.NPURunConfig', i[0])
            if (mode != None):
                findlist = list(set(re.compile('(npu_distributed_optimizer_wrapper)|(npu_allreduce)')
                                    .findall(i[0])))
                if len(findlist) == 0:
                    advisor.work('(npu_distributed_optimizer_wrapper)|(npu_allreduce)', 'estimator data parallel',
                                 0, "TF Training script", result)
                else:
                    if len(findlist) != 2:
                        advisor.work('(npu_distributed_optimizer_wrapper)|(npu_allreduce)', 'estimator data parallel',
                                     2, i[1], result)
            mode = re.search('sess.run', i[0])
            if (mode != None):
                findlist = list(set(re.compile('(broadcast)|(npu_distributed_optimizer_wrapper)|(npu_allreduce)')
                                    .findall(i[0])))
                if len(findlist) == 0:
                    advisor.work('(broadcast)|(npu_distributed_optimizer_wrapper)|(npu_allreduce)'
                                 , 'sess.run data parallel', 0, "TF Training script", result)
                else:
                    if len(findlist) != 3:
                        advisor.work('(broadcast)|(npu_distributed_optimizer_wrapper)|(npu_allreduce)'
                                     , 'sess.run data parallel', 2, i[1], result)

    # ----------------------------------------------Graph------------------------------------------------------
    data = processed_data('graph', data_path)
    dataTF = find_comparison.getprodata('TF', data)
    databsg = find_comparison.getprodata('BeforeSubGraph', data)
    join_TFgraphdata = joindata(dataTF)
    join_bsggraphdata = joindata(databsg)

    # 如果precision_mode不等于allow_mix_precision，给出优化建议
    advisor = Advisor('graph', join_TFgraphdata)
    advisor.work('precision_mode: (?!allow_mix_precision)', "Mixing precision2", 1, "DUMP graph", result)

    findlist = list(set(re.compile('(name: "MakeIterator")|(name: "IteratorV2")|(name: "Dataset")').findall(join_bsggraphdata)))
    if len(findlist) == 3:
        # 网络没有使用Dataset队列模式, 请尝试修改为Dataset接口模式,MakeIterator&&IteratorV2&&Dataset
        sadvisor.work('Dataset', 'The Dataset interface', 0, "BeforeSubGraph* graph", result)
        # Getnext没有下沉，请使能预处理下沉enable_data_pre_proc=True,MakeIterator&&IteratorV2;无enable_data_pre_proc
        if re.search('enable_data_pre_proc\s*=\s*True', join_scriptdata) == None:
            sadvisor.work('make_one_shot_iterator', 'Gennext sinking', 0, "BeforeSubGraph* graph", result)

    if data != []:
        for i in data:
            advisor = Advisor('graph', i[0])
            # 整网在使用debug模式，建议关闭op_debug_level
            advisor.work('op_debug_level(.*)1', "Overflow detection interface2", 1, i[1], result)
            # _iterations_per_loop是否大于1，如果等于1给出建议
            if re.search(r"(?<=iterations_per_loop:\s)\d+", i[0]):
                if re.search(r"(?<=iterations_per_loop:\s)\d+", i[0]).group() == '1':
                    buildrealresult.build_simpleresult(buildrealresult.get_r("CSR"),
                                                       'Circular sinking', i[1], result)

    # ----------------------------------------------plog---------------------------------------------------------------
    data = processed_data('plog', data_path)
    if data != []:
        advisor = Advisor('plog', data[0][0])
        # EVENT级别的plog中不包含“Use fusion library value”
        advisor.work('Use fusion library value', 'Gradient segmentation', '[\[]EVENT[\]]', data[0][1], result)

    # ----------------------------------------------scan profiling-----------------------------------------------------
    data = processed_data('profiling', data_path)
    if data != []:
        datas = find_comparison.getprodata('op_summary', data)[0]
        datast = find_comparison.getprodata('op_statistic', data)[0]

        # 扫描profiling数据，blockdim=1的耗时Top10范围内的算子罗列出来
        top = []
        j = 0
        for i in range(0, len(datas)-1):
            if(datas.iloc[i]["Block Dim"]==1 and datas.iloc[i]["Task Type"]=="AI_CORE"):
                top.append(datas.iloc[i])
        top = sorted(top, key=(lambda x: x["Task Duration(us)"]), reverse=True)
        top10 = top[:10]
        buildrealresult.build_protableresult(top10, 'Aicore operator',
                                             ['Op Name', 'Input Shapes', 'Output Shapes'], result)

        # 用户设置Transpose算子白名单
        tranlist = []
        for i in range(0, len(datast)-1):
            if(datast.iloc[i]['Core Type']=='AI_CORE' and datast.iloc[i]["Total Time(us)"] <= 1000
                    and datast.iloc[i]["Ratio(%)"] <= 1):
                tranlist.append(datast.iloc[i])
        buildrealresult.build_protableresult(tranlist, 'whitelist', ['OP Type', 'Total Time(us)', 'Ratio(%)'], result)
        
        # 算子输入dtype是INT64
        intlist = []
        for i in range(0, len(datas)-1):
            if(datas.iloc[i]['Task Type']=='AI_CPU'):
                if re.search('INT64', datas.iloc[i]['Input Data Types']) != None:
                    intlist.append(datas.iloc[i])
        buildrealresult.build_protableresult(intlist, 'INT64', ['OP Type', 'Task Type', 'Input Data Types'], result)

        # 扫描训练脚本中是否使用了dynamic_rnn接口;AICPU算子名称StackPushV2或者StackPopV2
        words = ['StackPushV2', 'StackPopV2']
        if find_comparison.isaicpu(words, datas) == 1:
            if scriptdata != []:
                for i in scriptdata:
                    advisor = Advisor('script', i[0])
                    advisor.work('(tf.)(nn.|contrib.rnn.)?(dynamic_rnn|bidirectional_dynamic_rnn|stack_bidirectional_'
                                'dynamic_rnn|static_bidirectional_rnn|static_rnn)',
                                'dynamic_rnn', 2, i[1], result)

        # 如果出现dropout/random_uniform/RandomUniform且是AICPU算子，给出优化建议
        intlist = []
        for i in range(0, len(datas)-1):
            if(datas.iloc[i]['Task Type']=='AI_CPU'):
                if re.search('(dropout)|(random_uniform)|(RandomUniform)', datas.iloc[i]['OP Type']) != None:
                    intlist.append(datas.iloc[i])
        buildrealresult.build_protableresult(intlist, 'dropout interface', ['OP Type', 'Task Type', 'Input Data Types'], result)

    # list type result
    if result.extend_result == []:
        result = buildrealresult.Result()
        result.class_type = buildrealresult.class_type['model']
        result.error_code = buildrealresult.error_code['optimized']
        result.summary = "Training scripts are well optimized"

    return result.generate()

if __name__ == "__main__":
    data_path = "../"
    ret = evaluate(data_path, "")
    print(ret)