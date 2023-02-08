import unittest
import os
import sys
import json
dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dirname + '/../src')
from model import evaluate


class TestTvmConst(unittest.TestCase):
    def setUp(self):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        self.data_path = dir_name + '/test_ops_file/4_leaky_relu_dsl'
        self.parameter = {'local': 'zh'}

    @classmethod
    def setUpClass(cls):
        print('CASE4'.center(100, '='))

    def test_evaluate(self):
        res = evaluate(self.data_path, self.parameter)
        res = json.loads(res)
        self.assertEqual(res['errorCode'], '0')
        self.assertEqual(res['classType'], '0')
        advice_no_list = list(map(lambda x: list(map(lambda y: y[4], x['value'])), res['extendResult']))
        advice_no_list_flat = [ele for row in advice_no_list for ele in row]
        self.assertTrue(4 in advice_no_list_flat)


if __name__ == '__main__':
    unittest.main()