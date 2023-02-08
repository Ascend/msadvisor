from work.flag import Type
import re
import buildrealresult
from advisor import Advisor
import find_comparison
from getnewdata import joindata

def graph(graphdata, join_scriptdata, sadvisor, result):
    dataTF = find_comparison.getprodata('TF', graphdata)
    databsg = find_comparison.getprodata('BeforeSubGraph', graphdata)
    join_TFgraphdata = joindata(dataTF)
    join_bsggraphdata = joindata(databsg)

    # ���precision_mode������allow_mix_precision�������Ż�����
    advisor = Advisor('graph', join_TFgraphdata)
    advisor.work('precision_mode: (?!allow_mix_precision)', "Mixing precision2", Type.EXIST, "DUMP graph", result)

    findlist = list(
        set(re.compile('(name: "MakeIterator")|(name: "IteratorV2")|(name: "Dataset")').findall(join_bsggraphdata)))
    if len(findlist) == 3:
        # ����û��ʹ��Dataset����ģʽ, �볢���޸�ΪDataset�ӿ�ģʽ,MakeIterator&&IteratorV2&&Dataset
        sadvisor.work('Dataset', 'The Dataset interface', Type.ABSENCE, "BeforeSubGraph* graph", result)
        # Getnextû���³�����ʹ��Ԥ�����³�enable_data_pre_proc=True,MakeIterator&&IteratorV2;��enable_data_pre_proc
        if re.search('enable_data_pre_proc\s*=\s*True', join_scriptdata) == None:
            sadvisor.work('make_one_shot_iterator', 'Gennext sinking', Type.ABSENCE, "BeforeSubGraph* graph", result)

    for i in graphdata:
        advisor = Advisor('graph', i[0])
        # ������ʹ��debugģʽ������ر�op_debug_level
        advisor.work('op_debug_level(.*)1', "Overflow detection interface2", Type.EXIST, i[1], result)
        # _iterations_per_loop�Ƿ����1���������1��������
        if re.search(r"(?<=iterations_per_loop:\s)\d+", i[0]):
            if re.search(r"(?<=iterations_per_loop:\s)\d+", i[0]).group() == '1':
                buildrealresult.build_simpleresult(buildrealresult.get_r("CSR"),
                                                   'Circular sinking', i[1], result)