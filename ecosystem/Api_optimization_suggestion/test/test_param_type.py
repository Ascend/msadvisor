import os
import sys
import json
import unittest
import shutil
from const import *
sys.path.append('../model/src/')
from run_model import evaluate, task_data


def make_param_type(params):
    default_params = \
        { 'ACL_VENC_BUF_SIZE_UINT32' : 5
        , 'ACL_VENC_MAX_BITRATE_UINT32' : 20
        , 'ACL_VENC_SRC_RATE_UINT32' : 0
        , 'ACL_VENC_RC_MODE_UINT32' : 1
        }
    default_params.update(params)
    return default_params

def format_python_test_data(params):
    ret = 'from enum import Enum\n\nclass aclvencChannelDescParamType(Enum):\n'
    if len(params) == 0:
        return ret + '    pass'
    for param, value in params.items():
        ret += f'    {param} = {value}\n'
    return ret


def make_test_data(params):
    project_path = os.path.join(fake_data_path, 'project')
    with open(os.path.join(project_path, 'constant.py'), 'w') as fp:
        fp.write(format_python_test_data(params))

    
class TestParamType(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_path = os.path.join(fake_data_path, 'project')
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

    def test_empty_params(self):
        make_test_data({})
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 3)
        params = [row[0] for row in extend_result[key_value]]
        self.assertTrue('ACL_VENC_BUF_SIZE_UINT32' in params)
        self.assertTrue('ACL_VENC_MAX_BITRATE_UINT32' in params)
        self.assertTrue('ACL_VENC_RC_MODE_UINT32' in params)

    def test_acl_venc_buf_size_uint32_out_bound(self):
        param = 'ACL_VENC_BUF_SIZE_UINT32'
        make_test_data(make_param_type({param : 1}))
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == param)
        self.assertTrue(extend_value[1] == task_data[param])
        self.assertTrue(extend_value[2] == 'constant.py Line:4')

    def test_acl_venc_max_bitrate_uint32_out_bound(self):
        param = 'ACL_VENC_MAX_BITRATE_UINT32'
        make_test_data(make_param_type({param : 1000000}))
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == param)
        self.assertTrue(extend_value[1] == task_data[param])
        self.assertTrue(extend_value[2] == 'constant.py Line:5')

    def test_acl_venc_max_bitrate_uint32_default(self):
        param = 'ACL_VENC_MAX_BITRATE_UINT32'
        make_test_data(make_param_type({param : 0}))
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == param)
        self.assertTrue('2000' in extend_value[1])
        self.assertTrue(extend_value[2] == 'constant.py Line:5')

    def test_acl_venc_src_rate_uint32_out_bound(self):
        param = 'ACL_VENC_SRC_RATE_UINT32'
        make_test_data(make_param_type({param : 250}))
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == param)
        self.assertTrue(extend_value[1] == task_data[param])
        self.assertTrue(extend_value[2] == 'constant.py Line:6')

    def test_acl_venc_rc_mode_uint32_default(self):
        param = 'ACL_VENC_RC_MODE_UINT32'
        make_test_data(make_param_type({param : 0}))
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == param)
        self.assertTrue('VBR' in extend_value[1])
        self.assertTrue(extend_value[2] == 'constant.py Line:7')

    def test_all_params(self):
        make_test_data(make_param_type({}))
        result = json.loads(evaluate(fake_data_path, 1))
        self._assert_well_optimized(result)


if __name__ == '__main__':
    unittest.main()