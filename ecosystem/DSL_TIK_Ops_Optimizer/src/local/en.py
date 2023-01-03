class Advice:
    """
    算子优化建议的文本
    """
    class AvoidLongRuntimeInterface:
        message = 'Avoid long runtime interfaces'
        vrec2vdiv = f"<{message}> The reciprocal calculation can be changed to the division calculation"
        vexp2neg = f"<{message}> Change the power of the exponent in the denominator to its negative"

    class AvoidExcessiveWrappingFunction:
        message = 'Avoid too much encapsulation of functions'

    class InlineConstant:
        message = 'tvm.const need not be defined separately'
        var_def = 'Delete the definition of this variable'
        var_use = 'Define tvm.const directly'

    class MultiCore:
        load_balance = 'Set block_num to a multiple of the number of AI cores to achieve load balancing'
        maximum = 'The block_num value cannot exceed the threshold of 65535'
        check_name = 'Check whether the variable or return value is a multiple of the number of AI cores'

    class DoubleBuffer:
        message = 'The performance gains of the double buffer should take into account the computation power of the ' \
                  'Vector, the size of the data, and the handling '

    class SyncInstruction:
        message = 'Enable the automatic insertion synchronization instruction when compiling. Ensure that the memory ' \
                  'address of data will not be trampled '
