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
statistic_data_file = 'acl_statistic_0_1_1.csv'


def format_profile_data(apis):
    ret = 'Name,Type,Start Time,Duration(us),Process ID,Thread ID\n'
    for api in apis:
        ret += f'{api},ACL_RTS,497845389418920,700.22,85078,85078\n'
    return ret

def format_statistic_data(apis):
    ret = 'Name,Type,Time(%),Time(us),Count,Avg(us),Min(us),Max(us),Process ID,Thread ID\n'
    for api in apis:
        ret += f'{api},ACL_MODEL,2.293195,3128.72,1,3128.72,3128.72,3128.72,85078,85078\n'
    return ret

def make_profile_data(apis):
    with open(os.path.join(summary_path, profile_data_file), 'w') as fp:
        fp.write(format_profile_data(apis))

def make_statistic_data(apis):
    with open(os.path.join(summary_path, statistic_data_file), 'w') as fp:
        fp.write(format_statistic_data(apis))


class TestMemoryManagement(unittest.TestCase):
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

    def test_no_memcpy(self):
        make_profile_data(['aclrtSetDevice'])
        make_statistic_data(['aclrtDeviceCanAccessPeer', 'aclrtDeviceEnablePeerAccess'])
        result = json.loads(evaluate(fake_data_path, 1))
        self._assert_well_optimized(result)

    def test_only_memcpy(self):
        make_profile_data(['aclrtMemcpy'])
        make_statistic_data(['aclrtMemcpy'])
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == 'aclrtMemcpy')
        self.assertTrue('aclrtDeviceCanAccessPeer' in extend_value[1] \
                        and 'aclrtDeviceEnablePeerAccess' in extend_value[1])
    
    def test_device_can_access_peer(self):
        make_profile_data(['aclrtDeviceCanAccessPeer', 'aclrtMemcpy'])
        make_statistic_data(['aclrtMemcpy'])
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == 'aclrtMemcpy')
        self.assertTrue('aclrtDeviceCanAccessPeer' not in extend_value[1] \
                        and 'aclrtDeviceEnablePeerAccess' in extend_value[1])

    def test_device_enable_peer_access(self):
        make_profile_data(['aclrtDeviceEnablePeerAccess', 'aclrtMemcpy'])
        make_statistic_data(['aclrtMemcpy'])
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == 'aclrtMemcpy')
        self.assertTrue('aclrtDeviceCanAccessPeer' in extend_value[1] \
                        and 'aclrtDeviceEnablePeerAccess' not in extend_value[1])

    def test_memory_both_check(self):
        make_profile_data(['aclrtDeviceCanAccessPeer', 'aclrtDeviceEnablePeerAccess', 'aclrtMemcpy'])
        make_statistic_data(['aclrtMemcpy'])
        result = json.loads(evaluate(fake_data_path, 1))
        self._assert_well_optimized(result)

    def test_memory_check_after_memcpy(self):
        make_profile_data([ 'aclrtMemcpy', 'aclrtDeviceCanAccessPeer', 'aclrtDeviceEnablePeerAccess'])
        make_statistic_data(['aclrtMemcpy'])
        result = json.loads(evaluate(fake_data_path, 1))
        extend_result = self._get_extend_result(result)
        self.assertTrue(key_value in extend_result and len(extend_result[key_value]) == 1)
        extend_value = extend_result[key_value][0]
        self.assertTrue(extend_value[0] == 'aclrtMemcpy')
        self.assertTrue('aclrtDeviceCanAccessPeer' in extend_value[1] \
                        and 'aclrtDeviceEnablePeerAccess' in extend_value[1])


if __name__ == '__main__':
    unittest.main()