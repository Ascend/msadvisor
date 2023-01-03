import re
from advisor import Advisor
from work.flag import Type

def script(sadvisor,join_scriptdata,scriptdata,result):
    # 训练脚本中不存在tf.data.Dataset字段
    sadvisor.work('tf.data.Dataset', 'The Dataset interface', Type.ABSENCE, "TF Training script", result)
    # 训练脚本中不包含NPURunConfig，不存在enable_data_pre_proc字段
    if re.search('NPURunConfig', join_scriptdata) == None:
        sadvisor.work('enable_data_pre_proc', 'Gennext sinking', Type.ABSENCE, "TF Training script", result)
    # 不存在allow_mix_precision字段
    sadvisor.work('allow_mix_precision', "Mixing precision", Type.ABSENCE, "TF Training script", result)
    # set_split_strategy_by_idx或者set_split_strategy_by_size没有设置，hcom_parallel未设置为True
    sadvisor.work('set_split_strategy_by', "Allreduce", Type.ABSENCE, "TF Training script", result)
    sadvisor.work('hcom_parallel\s*=\s*True', "Allreduce", Type.ABSENCE, "TF Training script", result)

    if scriptdata != []:
        for i in scriptdata:
            advisor = Advisor('script', i[0])
            # 扫描训练脚本中是否使用了溢出检测API
            advisor.work(
                '(tf.)(debugging.|math.|compat.v1.debugging.|compat.v1.|compat.v1.math.)?(is_finite|is_nan|is_inf)',
                'loss scale', Type.LINENUM, i[1], result)
            # # 没有npu_ops.dropout但训练脚本中存在dropout字段输出建议，有npu_ops.dropout则不提建议
            if re.search('npu_ops.dropout', i[0]) == None:
              advisor.work('(?<!npu_ops.)dropout', "dropout interface", Type.LINENUM, i[1], result)
            # 使用了原始小算子自定义gelu函数
            advisor.work('def gelu[(]', "gelu interface", Type.LINENUM, i[1], result)
            # 存在mix_compile_mode字段
            advisor.work('mix_compile_mode', "mixed calculation", Type.LINENUM, i[1], result)
            # 存在tf.boolean_mask字段
            advisor.work('tf.boolean_mask', "interface1", Type.LINENUM, i[1], result)
            # LSTM接口
            advisor.work(
                '(tf.)(nn.rnn_cell.BasicLSTMCell|contrib.rnn.LayerNormBasicLSTMCell|contrib.rnn.BasicLSTMCell)',
                'LSTM', Type.LINENUM, i[1], result)
            # keras框架 tensorflow.keras，xxx.compile(xxx,loss="categorical_crossentropy", metrics=["accuracy"])
            if re.search('tensorflow.keras', i[0]) != None:
                advisor.work(['\.compile[(]([\s\S]*),\n\s*[)]', 'metrics\s*=\s*[\[]"accuracy"[\]]',
                              'loss\s*=\s*"categorical_crossentropy"'], 'interface2', Type.BLOCK, i[1], result)
            # Estimator,sess.run,Keras三种模式
            if re.search('tensorflow.keras', i[0]) != None:
                mode = re.search('\.compile[(]', i[0])
                if (mode != None):
                    findlist = list(
                        set(re.compile('(npu_distributed_optimizer_wrapper)|(NPUBroadcastGlobalVariablesCallback)'
                                       ).findall(i[0])))
                    if len(findlist) == 0:
                        advisor.work('(npu_distributed_optimizer_wrapper)|(NPUBroadcastGlobalVariablesCallback)',
                                     'Keras data parallel', Type.ABSENCE, "TF Training script", result)
                    else:
                        if len(findlist) != 2:
                            advisor.work('(npu_distributed_optimizer_wrapper)|(NPUBroadcastGlobalVariablesCallback)',
                                         'Keras data parallel', Type.LINENUM, i[1], result)
            mode = re.search('(.*).estimator.NPURunConfig', i[0])
            if (mode != None):
                findlist = list(set(re.compile('(npu_distributed_optimizer_wrapper)|(npu_allreduce)')
                                    .findall(i[0])))
                if len(findlist) == 0:
                    advisor.work('(npu_distributed_optimizer_wrapper)|(npu_allreduce)', 'estimator data parallel',
                                 Type.ABSENCE, "TF Training script", result)
                else:
                    if len(findlist) != 2:
                        advisor.work('(npu_distributed_optimizer_wrapper)|(npu_allreduce)', 'estimator data parallel',
                                     Type.LINENUM, i[1], result)
            mode = re.search('sess.run', i[0])
            if (mode != None):
                findlist = list(set(re.compile('(broadcast)|(npu_distributed_optimizer_wrapper)|(npu_allreduce)')
                                    .findall(i[0])))
                if len(findlist) == 0:
                    advisor.work('(broadcast)|(npu_distributed_optimizer_wrapper)|(npu_allreduce)'
                                 , 'sess.run data parallel', Type.ABSENCE, "TF Training script", result)
                else:
                    if len(findlist) != 3:
                        advisor.work('(broadcast)|(npu_distributed_optimizer_wrapper)|(npu_allreduce)'
                                     , 'sess.run data parallel', Type.LINENUM, i[1], result)