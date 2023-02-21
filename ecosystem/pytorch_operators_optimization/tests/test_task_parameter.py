import sys
import os
import unittest
import json

sys.path.append('src')

from model import evaluate
from misc import set_up_helper


class TestTaskParameter(unittest.TestCase):
    def setUp(self) -> None:
        pos_content = (
            'learning_rate=0.3' + os.linesep +
            'batch_size=1' + os.linesep
        )
        set_up_helper('parameter_pos_1', pos_content, 'sh')
        pos_content = (
            'learning_rate=0.3' + os.linesep +
            'bs=1' + os.linesep
        )
        set_up_helper('parameter_pos_2', pos_content, 'sh')
        pos_content = (
            'lr=0.3' + os.linesep +
            'batch_size=1' + os.linesep
        )
        set_up_helper('parameter_pos_3', pos_content, 'sh')
        pos_content = (
            'lr=0.3' + os.linesep +
            'bs=1' + os.linesep
        )
        set_up_helper('parameter_pos_4', pos_content, 'sh')
        neg_content = (
            'test=1' + os.linesep +
            '' + os.linesep
        )
        set_up_helper('parameter_neg', neg_content, 'sh')

    def tearDown(self) -> None:
        os.system('rm -rf ../data/project/test_task_parameter_*')

    def test_task_parameter_pos_1(self):
        res_s = evaluate('../data/project/test_task_parameter_pos_1', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')

    def test_task_parameter_pos_2(self):
        res_s = evaluate('../data/project/test_task_parameter_pos_2', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any('Batch_Size' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_parameter_pos_3(self):
        res_s = evaluate('../data/project/test_task_parameter_pos_3', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any('Batch_Size' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_parameter_pos_4(self):
        res_s = evaluate('../data/project/test_task_parameter_pos_4', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any('Batch_Size' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_parameter_neg(self):
        res_s = evaluate('../data/project/test_task_parameter_neg', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '1')
        self.assertFalse(any('Batch_Size' in item for r in res['extendResult'] for v in r['value'] for item in v))


if __name__ == "__main__":
    unittest.main()
