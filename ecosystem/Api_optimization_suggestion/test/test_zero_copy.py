import os
import sys
import json
import shutil
import unittest
from const import *
sys.path.append('../model/src/')
from run_model import evaluate


class Malloc:
    def __init__(self, src: str):
        self._src = src
    
    def __str__(self) -> str:
        return f'acldvppMalloc({self._src}, size);\n'


class Memcpy:
    def __init__(self, src: str, dst: str):
        self._src = src
        self._dst = dst
    
    def __str__(self) -> str:
        return f'aclrtMemcpy({self._dst}, destMax, {self._src}, count, kind);\n'


project_path = os.path.join(fake_data_path, 'project')
fake_cpp_file = 'fake_data.cpp'

def format_cpp_data(apis):
    return ''.join(map(str, apis))

def format_default_param_type(params):
    ret = 'enum aclvencChannelDescParamType {\n'
    if len(params) == 0:
        return ret + '    pass'
    for param, value in params.items():
        ret += f'    {param} = {value}\n'
    return ret + '}\n'

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


class TestZeroCopy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(project_path):
            os.makedirs(project_path)

    @classmethod
    def tearDownClass(cls):
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

    def test_no_malloc_no_memcpy(self):
        make_cpp_data([])
        result = json.loads(evaluate(fake_data_path, 1))
        self._assert_well_optimized(result)

    def test_only_malloc(self):
        make_cpp_data([Malloc('dvpp')])
        result = json.loads(evaluate(fake_data_path, 1))
        self._assert_well_optimized(result)

    def test_memcpy_from_dvpp(self):
        make_cpp_data([Malloc('dvpp'), Memcpy('dvpp', 'dst')])
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == 'aclrtMemcpy')
        self.assertTrue('not need to copy the data' in extend_value[1])
        self.assertTrue(extend_value[2] == f'{fake_cpp_file} Line:8')

    def test_memcpy_to_dvpp(self):
        make_cpp_data([Malloc('dvpp'), Memcpy('src', 'dvpp')])
        result = json.loads(evaluate(fake_data_path, 1))
        self._assert_well_optimized(result)

    def test_memcpy_from_non_dvpp(self):
        make_cpp_data([Malloc('dvpp'), Memcpy('src', 'dst')])
        result = json.loads(evaluate(fake_data_path, 1))
        self._assert_well_optimized(result)


if __name__ == '__main__':
    unittest.main()