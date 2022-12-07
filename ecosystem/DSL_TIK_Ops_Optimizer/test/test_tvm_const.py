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
        self.data_path = dir_name + '/test_ops_file/4_leaky_relu_dsl.py'
        self.parameter = {'local': 'zh'}

    @classmethod
    def setUpClass(cls):
        print('CASE4'.center(100, '='))

    def test_evaluate(self):
        res = evaluate(self.data_path, self.parameter)
        res = json.loads(res)
        self.assertEqual(res['errorCode'], '0')
        self.assertEqual(res['classType'], '0')


if __name__ == '__main__':
    unittest.main()