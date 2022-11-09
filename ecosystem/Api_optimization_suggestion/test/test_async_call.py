import os
import sys
import json
import shutil
import unittest
from const import *
sys.path.append('../model/src/')
from run_model import evaluate


summary_path = os.path.join(fake_data_path, 'profiler', 'PROF_fake', 'device_0', 'summary')
statistic_data_file = 'acl_statistic_0_1_1.csv'

def format_statistic_data(apis):
    ret = 'Name,Type,Time(%),Time(us),Count,Avg(us),Min(us),Max(us),Process ID,Thread ID\n'
    for api in apis:
        ret += f'{api},ACL_MODEL,2.293195,3128.72,1,3128.72,3128.72,3128.72,85078,85078\n'
    return ret

def make_statistic_data(apis):
    with open(os.path.join(summary_path, statistic_data_file), 'w') as fp:
        fp.write(format_statistic_data(apis))


class TestAsyncCall(unittest.TestCase):
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

    def test_no_execute_async(self):
        make_statistic_data(['aclrtDeviceCanAccessPeer', 'aclrtDeviceEnablePeerAccess', 'aclrtMemcpy'])
        result = json.loads(evaluate(fake_data_path, 1))
        self._assert_well_optimized(result)

    def test_only_synchronize_stream(self):
        make_statistic_data(['aclrtSynchronizeStream', 'aclrtDeviceEnablePeerAccess', 'aclrtMemcpy'])
        result = json.loads(evaluate(fake_data_path, 1))
        self._assert_well_optimized(result)

    def test_only_execute_async(self):
        make_statistic_data(['aclmdlExecuteAsync', 'aclmdlExecuteAsync', 'aclrtMemcpy'])
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == 'aclmdlExecuteAsync')
        self.assertTrue('aclrtSynchronizeStream' in extend_value[1])
        self.assertTrue(extend_value[2] == 'Line:1, 2')

    def test_async_with_synchronize(self):
        make_statistic_data(['aclmdlExecuteAsync', 'aclrtSynchronizeStream', 'aclrtDeviceEnablePeerAccess'])
        result = json.loads(evaluate(fake_data_path, 1))
        self._assert_well_optimized(result)

    def test_async_far_from_synchronize(self):
        make_statistic_data(['aclmdlExecuteAsync', 'aclrtDeviceEnablePeerAccess', 'aclrtSynchronizeStream'])
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == 'aclmdlExecuteAsync')
        self.assertTrue('aclrtSynchronizeStream' in extend_value[1])
        self.assertTrue(extend_value[2] == 'Line:1')

    def test_multi_async_with_synchronize(self):
        make_statistic_data(['aclmdlExecuteAsync', 'aclmdlExecuteAsync', 'aclrtSynchronizeStream',
                             'aclmdlExecuteAsync', 'aclrtSynchronizeStream'])
        result = json.loads(evaluate(fake_data_path, 1))
        self._assert_well_optimized(result)


if __name__ == '__main__':
    unittest.main()