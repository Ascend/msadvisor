class Advice:
    """
    算子优化建议的文本
    """
    class AvoidLongRuntimeInterface:
        message = '避免调用运行时间过程的接口'
        vrec2vdiv = f"<{message}> 可将取倒数计算改为除法计算"
        vexp2neg = f"<{message}> 可将分母中的指数运算中幂改为其相反数"

    class AvoidExcessiveWrappingFunction:
        message = '避免函数过多封装'

    class InlineConstant:
        message = 'tvm.const不必单独定义'
        var_def = '可删除该变量的定义'
        var_use = '修改为tvm.const直接定义的形式'

    class MultiCore:
        load_balance = '设置block_num建议为AI Core数目的倍数，以达到负载均衡'
        maximum = '设置block_num不得超过阈值65535'
        check_name = '建议检查变量或返回值是否为AI Core数目的倍数'

    class DoubleBuffer:
        message = 'double buffer的性能收益需综合考虑Vector算力、数据量大小、搬运与计算时间占比等多种因素'

    class SyncInstruction:
        message = '开启编译时自动插入同步指令请确保数据的内存地址不会发生踩踏的问题'
