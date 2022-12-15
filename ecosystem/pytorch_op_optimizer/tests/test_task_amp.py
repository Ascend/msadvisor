import sys
import os
import unittest
import json

sys.path.append('src')

from model import evaluate
from misc import set_up_helper


class TestTaskAmp(unittest.TestCase):
    def setUp(self) -> None:
        pos_content_1 = (
            "model, optimizer = amp.initialize(model, optimizer, combine_grad=False)" + os.linesep +
            "model, optimizer = amp.initialize(" + os.linesep +
            "    model, optimizer, opt_level='O2', loss_scale=32.0)" + os.linesep
        )
        set_up_helper('amp_pos_1', pos_content_1)
        pos_content_2 = (
            "model, optimizer = amp.initialize(model, optimizer, opt_level='O2', loss_scale=32.0)" + os.linesep +
            "model, optimizer = amp.initialize(" + os.linesep +
            "    model, optimizer, opt_level='O2', loss_scale=32.0)" + os.linesep
        )
        set_up_helper('amp_pos_2', pos_content_2)
        neg_content = (
            "model, optimizer = amp.initialize(model, optimizer, combine_grad = True)" + os.linesep +
            "model, optimizer = amp.initialize(model, optimizer, combine_grad=True)" + os.linesep +
            'model, optimizer = amp.initialize(' + os.linesep +
            '    model, optimizer, combine_grad=True)' + os.linesep
        )
        set_up_helper('amp_neg', neg_content)

    def tearDown(self) -> None:
        os.system('rm -rf ../data/project/test_task_amp_*')

    def test_task_amp_pos_1(self):
        res_s = evaluate('../data/project/test_task_amp_pos_1', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any('amp' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_amp_pos_2(self):
        res_s = evaluate('../data/project/test_task_amp_pos_2', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any('amp' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_amp_neg(self):
        res_s = evaluate('../data/project/test_task_amp_neg', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '1')
        self.assertFalse(any('amp' in item for r in res['extendResult'] for v in r['value'] for item in v))


if __name__ == "__main__":
    unittest.main()
