import re
import buildrealresult
from advisor import Advisor
import find_comparison
from work.flag import Type

def profiling(prodata,scriptdata,result):
    if prodata != []:
        datas = find_comparison.getprodata('op_summary', prodata)[0]
        datast = find_comparison.getprodata('op_statistic', prodata)[0]

        # ɨ��profiling���ݣ�blockdim=1�ĺ�ʱTop10��Χ�ڵ��������г���
        top = []
        for i in range(0, len(datas) - 1):
            if (datas.iloc[i]["Block Dim"] == 1 and datas.iloc[i]["Task Type"] == "AI_CORE"):
                top.append(datas.iloc[i])
        top = sorted(top, key=(lambda x: x["Task Duration(us)"]), reverse=True)
        top10 = top[:10]
        buildrealresult.build_protableresult(top10, 'Aicore operator',
                                             ['Op Name', 'Input Shapes', 'Output Shapes'], result)

        # �û�����Transpose���Ӱ�����
        tranlist = []
        for i in range(0, len(datast) - 1):
            if (datast.iloc[i]['Core Type'] == 'AI_CORE' and datast.iloc[i]["Total Time(us)"] <= 1000
                    and datast.iloc[i]["Ratio(%)"] <= 1):
                tranlist.append(datast.iloc[i])
        buildrealresult.build_protableresult(tranlist, 'whitelist', ['OP Type', 'Total Time(us)', 'Ratio(%)'], result)

        # ��������dtype��INT64
        intlist = []
        for i in range(0, len(datas) - 1):
            if (datas.iloc[i]['Task Type'] == 'AI_CPU'):
                if re.search('INT64', datas.iloc[i]['Input Data Types']) != None:
                    intlist.append(datas.iloc[i])
        buildrealresult.build_protableresult(intlist, 'INT64', ['OP Type', 'Task Type', 'Input Data Types'], result)

        # ɨ��ѵ���ű����Ƿ�ʹ����dynamic_rnn�ӿ�;AICPU��������StackPushV2����StackPopV2
        words = ['StackPushV2', 'StackPopV2']
        if find_comparison.isaicpu(words, datas) == 1:
            if scriptdata != []:
                for i in scriptdata:
                    advisor = Advisor('script', i[0])
                    advisor.work('(tf.)(nn.|contrib.rnn.)?(dynamic_rnn|bidirectional_dynamic_rnn|stack_bidirectional_'
                                 'dynamic_rnn|static_bidirectional_rnn|static_rnn)',
                                 'dynamic_rnn', Type.LINENUM, i[1], result)

        # �������dropout/random_uniform/RandomUniform����AICPU���ӣ������Ż�����
        intlist = []
        for i in range(0, len(datas) - 1):
            if (datas.iloc[i]['Task Type'] == 'AI_CPU'):
                if re.search('(dropout)|(random_uniform)|(RandomUniform)', datas.iloc[i]['OP Type']) != None:
                    intlist.append(datas.iloc[i])
        buildrealresult.build_protableresult(intlist, 'dropout interface', ['OP Type', 'Task Type', 'Input Data Types'],
                                             result)