import sys
import os
import unittest
import json

sys.path.append('src')

from model import evaluate
from misc import set_up_helper


class TestTaskOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        pos_content_1 = (
            'optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)' + os.linesep
        )
        set_up_helper('optimizer_pos_1', pos_content_1)
        pos_content_2 = (
            'optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, momentum=args.momentum)' + os.linesep
        )
        set_up_helper('optimizer_pos_2', pos_content_2)
        neg_content = (
            'optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, momentum=args.momentum)' + os.linesep +
            'optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, momentum=args.momentum)' + os.linesep
        )
        set_up_helper('optimizer_neg', neg_content)

    def tearDown(self) -> None:
        os.system('rm -rf ../data/project/test_task_optimizer_*')

    def test_task_optimizer_pos_1(self):
        res_s = evaluate('../data/project/test_task_optimizer_pos_1', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any('NpuFusedSGD' in item for r in res['extendResult'] for v in r['value'] for item in v))
        self.assertFalse(any('NpuFusedAdam' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_optimizer_pos_2(self):
        res_s = evaluate('../data/project/test_task_optimizer_pos_2', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertFalse(any('NpuFusedSGD' in item for r in res['extendResult'] for v in r['value'] for item in v))
        self.assertTrue(any('NpuFusedAdam' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_optimizer_neg(self):
        res_s = evaluate('../data/project/test_task_optimizer_neg', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '1')
        self.assertFalse(any('NpuFusedSGD' in item for r in res['extendResult'] for v in r['value'] for item in v))
        self.assertFalse(any('NpuFusedAdam' in item for r in res['extendResult'] for v in r['value'] for item in v))


if __name__ == "__main__":
    unittest.main()
