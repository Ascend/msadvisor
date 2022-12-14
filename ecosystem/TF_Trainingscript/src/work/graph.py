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

    # 如果precision_mode不等于allow_mix_precision，给出优化建议
    advisor = Advisor('graph', join_TFgraphdata)
    advisor.work('precision_mode: (?!allow_mix_precision)', "Mixing precision2", Type.EXIST, "DUMP graph", result)

    findlist = list(
        set(re.compile('(name: "MakeIterator")|(name: "IteratorV2")|(name: "Dataset")').findall(join_bsggraphdata)))
    if len(findlist) == 3:
        # 网络没有使用Dataset队列模式, 请尝试修改为Dataset接口模式,MakeIterator&&IteratorV2&&Dataset
        sadvisor.work('Dataset', 'The Dataset interface', Type.ABSENCE, "BeforeSubGraph* graph", result)
        # Getnext没有下沉，请使能预处理下沉enable_data_pre_proc=True,MakeIterator&&IteratorV2;无enable_data_pre_proc
        if re.search('enable_data_pre_proc\s*=\s*True', join_scriptdata) == None:
            sadvisor.work('make_one_shot_iterator', 'Gennext sinking', Type.ABSENCE, "BeforeSubGraph* graph", result)

    for i in graphdata:
        advisor = Advisor('graph', i[0])
        # 整网在使用debug模式，建议关闭op_debug_level
        advisor.work('op_debug_level(.*)1', "Overflow detection interface2", Type.EXIST, i[1], result)
        # _iterations_per_loop是否大于1，如果等于1给出建议
        if re.search(r"(?<=iterations_per_loop:\s)\d+", i[0]):
            if re.search(r"(?<=iterations_per_loop:\s)\d+", i[0]).group() == '1':
                buildrealresult.build_simpleresult(buildrealresult.get_r("CSR"),
                                                   'Circular sinking', i[1], result)