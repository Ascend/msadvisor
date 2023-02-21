import sys
import os
import unittest
import json

sys.path.append('src')

from model import evaluate
from misc import set_up_helper


class TestTaskTaskset(unittest.TestCase):
    def setUp(self) -> None:
        pos_content = (
            'taskset' + os.linesep +
            '' + os.linesep
        )
        set_up_helper('taskset_pos', pos_content, 'sh')
        neg_content = (
            '# taskset' + os.linesep +
            '' + os.linesep
        )
        set_up_helper('taskset_neg', neg_content, 'sh')

    def tearDown(self) -> None:
        os.system('rm -rf ../data/project/test_task_taskset_*')

    def test_task_taskset_pos(self):
        res_s = evaluate('../data/project/test_task_taskset_pos', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any('taskset' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_taskset_neg(self):
        res_s = evaluate('../data/project/test_task_taskset_neg', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '1')
        self.assertFalse(any('taskset' in item for r in res['extendResult'] for v in r['value'] for item in v))


if __name__ == "__main__":
    unittest.main()
