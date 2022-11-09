import os
import sys
import json
import shutil
import unittest
from const import *
sys.path.append('../model/src/')
from run_model import evaluate


summary_path = os.path.join(fake_data_path, 'profiler', 'PROF_fake', 'device_0', 'summary')
profile_data_file = 'acl_0_1_1.csv'

def format_profile_data(apis):
    ret = 'Name,Type,Start Time,Duration(us),Process ID,Thread ID\n'
    for api in apis:
        ret += f'{api},ACL_RTS,497845389418920,700.22,85078,85078\n'
    return ret

def make_profile_data(apis):
    with open(os.path.join(summary_path, profile_data_file), 'w') as fp:
        fp.write(format_profile_data(apis))


class TestContextManagement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)

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

    def test_no_stream(self):
        make_profile_data([])
        result = json.loads(evaluate(fake_data_path, 1))
        self._assert_well_optimized(result)

    def test_create_many_streams(self):
        make_profile_data(['aclrtCreateStream'] * 2048)
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == 'aclrtCreateStream')
        self.assertTrue('1024' in extend_value[1] and '2048' in extend_value[1])

    def test_create_many_contexts(self):
        make_profile_data(['aclrtCreateContext'] * 2048)
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == 'aclrtCreateStream')
        self.assertTrue('1024' in extend_value[1] and '2049' in extend_value[1])

    def test_destroy_after_create(self):
        make_profile_data(['aclrtCreateContext'] * 2048 + ['aclrtDestroyStream'] * 1024)
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == 'aclrtCreateStream')
        self.assertTrue('1024' in extend_value[1] and '1025' in extend_value[1])

    def test_destroy_before_create(self):
        make_profile_data(['aclrtDestroyStream'] * 1024 + ['aclrtCreateContext'] * 2048)
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == 'aclrtCreateStream')
        self.assertTrue('1024' in extend_value[1] and '2049' in extend_value[1])


if __name__ == '__main__':
    unittest.main()