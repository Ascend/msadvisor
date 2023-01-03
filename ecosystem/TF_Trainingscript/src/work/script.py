import re
from advisor import Advisor
from work.flag import Type

def script(sadvisor,join_scriptdata,scriptdata,result):
    # ѵ���ű��в�����tf.data.Dataset�ֶ�
    sadvisor.work('tf.data.Dataset', 'The Dataset interface', Type.ABSENCE, "TF Training script", result)
    # ѵ���ű��в�����NPURunConfig��������enable_data_pre_proc�ֶ�
    if re.search('NPURunConfig', join_scriptdata) == None:
        sadvisor.work('enable_data_pre_proc', 'Gennext sinking', Type.ABSENCE, "TF Training script", result)
    # ������allow_mix_precision�ֶ�
    sadvisor.work('allow_mix_precision', "Mixing precision", Type.ABSENCE, "TF Training script", result)
    # set_split_strategy_by_idx����set_split_strategy_by_sizeû�����ã�hcom_parallelδ����ΪTrue
    sadvisor.work('set_split_strategy_by', "Allreduce", Type.ABSENCE, "TF Training script", result)
    sadvisor.work('hcom_parallel\s*=\s*True', "Allreduce", Type.ABSENCE, "TF Training script", result)

    if scriptdata != []:
        for i in scriptdata:
            advisor = Advisor('script', i[0])
            # ɨ��ѵ���ű����Ƿ�ʹ����������API
            advisor.work(
                '(tf.)(debugging.|math.|compat.v1.debugging.|compat.v1.|compat.v1.math.)?(is_finite|is_nan|is_inf)',
                'loss scale', Type.LINENUM, i[1], result)
            # # û��npu_ops.dropout��ѵ���ű��д���dropout�ֶ�������飬��npu_ops.dropout���Ὠ��
            if re.search('npu_ops.dropout', i[0]) == None:
              advisor.work('(?<!npu_ops.)dropout', "dropout interface", Type.LINENUM, i[1], result)
            # ʹ����ԭʼС�����Զ���gelu����
            advisor.work('def gelu[(]', "gelu interface", Type.LINENUM, i[1], result)
            # ����mix_compile_mode�ֶ�
            advisor.work('mix_compile_mode', "mixed calculation", Type.LINENUM, i[1], result)
            # ����tf.boolean_mask�ֶ�
            advisor.work('tf.boolean_mask', "interface1", Type.LINENUM, i[1], result)
            # LSTM�ӿ�
            advisor.work(
                '(tf.)(nn.rnn_cell.BasicLSTMCell|contrib.rnn.LayerNormBasicLSTMCell|contrib.rnn.BasicLSTMCell)',
                'LSTM', Type.LINENUM, i[1], result)
            # keras��� tensorflow.keras��xxx.compile(xxx,loss="categorical_crossentropy", metrics=["accuracy"])
            if re.search('tensorflow.keras', i[0]) != None:
                advisor.work(['\.compile[(]([\s\S]*),\n\s*[)]', 'metrics\s*=\s*[\[]"accuracy"[\]]',
                              'loss\s*=\s*"categorical_crossentropy"'], 'interface2', Type.BLOCK, i[1], result)
            # Estimator,sess.run,Keras����ģʽ
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