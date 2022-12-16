from TransData_identify_optimization_advice import evaluate
import os


# 正常测试flag为True，异常测试为False
def test_fun(data_path, onnx_name, language='English', flag=True):
    parameter = {'onnx_name': onnx_name, 'language': language}
    if flag:
        print(f"{onnx_name}对应的优化建议如下：")
        ret = evaluate(data_path, parameter)
        print(ret)
    else:
        print("异常打印如下：")
        evaluate(data_path, parameter)
    print("--------------------------------------------------------------")


if __name__ == "__main__":
    path = os.path.abspath(os.path.join(__file__, "../.."))
    # 所有正常测试用例都在data/data下
    data_path = os.path.join(path, 'data/data')

    # 测试场景1 test_4D_5HD，且输入中文的优化建议。
    test_fun(data_path, 'test_4D_5HD.onnx', 'Chinese', True)

    # 测试场景2 test_ND_HZ，且输入英文的优化建议。
    test_fun(data_path, 'test_ND_NZ.onnx', 'English', True)

    # 测试场景3 test_reshape，且输入英文的优化建议。
    test_fun(data_path, 'test_reshape.onnx')

    # 测试场景4 test_conv
    test_fun(data_path, 'test_conv.onnx')

    # 测试场景5 test_matmul
    test_fun(data_path, 'test_matmul.onnx')

    # 测试场景6 test_conv_matmul
    test_fun(data_path, 'test_conv_matmul.onnx')

    # 测试场景7 test_no_transdata
    test_fun(data_path, 'test_no_transdata.onnx')

    # 异常测试8：onnx_name不正确，test_abc.onnx不存在
    test_fun(data_path, 'test_abc.onnx', 'English', False)

    # 异常测试9：data_path不正确,path直接放data目录，而不是data/data
    test_fun(path, 'test_4D_5HD.onnx', 'English', False)

    # 异常测试10：data1中只有PROF文件，没有onnx图
    error_path = os.path.join(path, 'data/data1')
    test_fun(error_path, 'test_4D_5HD.onnx', 'English', False)

    # 异常测试11：data2的PROF中中没有device文件夹
    error_path = os.path.join(path, 'data/data2')
    test_fun(error_path, 'test_4D_5HD.onnx', 'English', False)

    # 异常测试12：data3/profiling中没有PROF文件夹
    error_path = os.path.join(path, 'data/data3')
    test_fun(error_path, 'test_4D_5HD.onnx', 'English', False)
