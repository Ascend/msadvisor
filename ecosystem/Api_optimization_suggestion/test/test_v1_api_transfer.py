import os
import sys
import json
import shutil
import unittest
from unittest import result
from const import *
sys.path.append('../model/src/')
from run_model import evaluate, V1_transformer


summary_path = os.path.join(fake_data_path, 'profiler', 'PROF_fake', 'device_0', 'summary')
statistic_data_file = 'acl_statistic_0_1_1.csv'
project_path = os.path.join(fake_data_path, 'project')
fake_cpp_file = 'fake_data.cpp'


class Interpolation:
    def __init__(self, mode: int):
        self._mode = mode

    def __str__(self) -> str:
        return f'acldvppSetResizeConfigInterpolation(1, {self._mode}, another_param);\n'

class SetChannelDescParam:
    def __str__(self) -> str:
        return f'aclvencSetChannelDescParam();\n'


def format_statistic_data(apis):
    ret = 'Name,Type,Time(%),Time(us),Count,Avg(us),Min(us),Max(us),Process ID,Thread ID\n'
    for api in apis:
        ret += f'{api},ACL_MODEL,2.293195,3128.72,1,3128.72,3128.72,3128.72,85078,85078\n'
    return ret


def make_statistic_data(apis):
    with open(os.path.join(summary_path, statistic_data_file), 'w') as fp:
        fp.write(format_statistic_data(apis))

def format_cpp_data(apis):
    return ''.join(map(str, apis))

def format_default_param_type(params):
    ret = 'enum aclvencChannelDescParamType {\n'
    if len(params) == 0:
        return ret + '    pass'
    for param, value in params.items():
        ret += f'    {param} = {value}\n'
    return ret + '};\n'

def make_cpp_data(apis):
    with open(os.path.join(project_path, fake_cpp_file), 'w') as fp:
        # Make a default param value file to avoid suggestions for default values
        fp.write(format_default_param_type(
            { 'ACL_VENC_BUF_SIZE_UINT32' : 5
            , 'ACL_VENC_MAX_BITRATE_UINT32' : 20
            , 'ACL_VENC_SRC_RATE_UINT32' : 0
            , 'ACL_VENC_RC_MODE_UINT32' : 1
            }))
        fp.write(format_cpp_data(apis))


class TestV1ApiTransfer(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        if not os.path.exists(project_path):
            os.makedirs(project_path)

    def tearDown(self):
        if os.path.isdir(fake_data_path):
            shutil.rmtree(fake_data_path)
    
    def _get_extend_result(self, result):
        self.assertTrue(key_class_type in result and result[key_class_type] == '0')
        self.assertTrue(key_error_code in result and result[key_error_code] == '0')
        self.assertTrue(key_summary in result and 'need to be optimized' in result[key_summary])
        self.assertTrue(key_extend_result in result and len(result[key_extend_result]) == 1)
        extend_result = result[key_extend_result][0]
        self.assertTrue(key_type in  extend_result and extend_result[key_type] == '1')
        return extend_result

    def _assert_well_optimized(self, result):
        self.assertTrue(key_class_type in result and result[key_class_type] == '0')
        self.assertTrue(key_error_code in result and result[key_error_code] == '1')
        self.assertTrue(key_summary in result and 'well optimized' in result[key_summary])
        self.assertTrue(key_extend_result in result and len(result[key_extend_result]) == 0)

    def test_no_v1_api(self):
        make_statistic_data([])
        result = json.loads(evaluate(fake_data_path, 1))
        self._assert_well_optimized(result)

    def test_each_v1_api(self):
        for api in V1_transformer:
            make_statistic_data([api])
            result = json.loads(evaluate(fake_data_path, 1))
            extend_result = self._get_extend_result(result)
            self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
            extend_value = extend_result[key_value][0]
            self.assertTrue(extend_value[0] == api)
            self.assertTrue(extend_value[1] == V1_transformer[api])
            self.assertTrue(extend_value[2] == 'Line:1')

    def test_all_apis(self):
        apis = list(V1_transformer.keys())
        make_statistic_data(apis)
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == len(apis))
        for index, extend_value in enumerate(extend_result[key_value]):
            self.assertTrue(extend_value[0] == apis[index])
            self.assertTrue(extend_value[1] == V1_transformer[apis[index]])
            self.assertTrue(extend_value[2] == f'Line:{index + 1}')

    def test_not_interpolation(self):
        make_cpp_data([])
        result = json.loads(evaluate(fake_data_path, 1))
        self._assert_well_optimized(result)

    def test_interpolation_default(self):
        make_cpp_data([Interpolation(0)])
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == 'acldvppSetResizeConfigInterpolation')
        self.assertTrue('0:(default)Bilinear algorithm' in extend_value[1])
        self.assertTrue(extend_value[2] == f'{fake_cpp_file} Line:7')

    def test_interpolation_not_support(self):
        make_cpp_data([Interpolation(3), Interpolation((4))])
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 2)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == 'acldvppSetResizeConfigInterpolation')
        self.assertTrue('only support 0:(default)Bilinear algorithm' in extend_value[1])
        self.assertTrue(extend_value[2] == f'{fake_cpp_file} Line:7')
        extend_value = extend_result[key_value][1]
        self.assertTrue(extend_value[0] == 'acldvppSetResizeConfigInterpolation')
        self.assertTrue('only support 0:(default)Bilinear algorithm' in extend_value[1])
        self.assertTrue(extend_value[2] == f'{fake_cpp_file} Line:8')

    def test_set_channel_desc_param(self):
        make_cpp_data([SetChannelDescParam()])
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == 'aclvencSetChannelDescParam')
        self.assertTrue('cannot use the set IP ratio function' in extend_value[1])
        self.assertTrue(extend_value[2] == f'{fake_cpp_file} Line:7')


if __name__ == '__main__':
    unittest.main()