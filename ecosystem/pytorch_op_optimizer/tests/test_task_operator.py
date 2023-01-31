import sys
import os
import unittest
import json

sys.path.append('src')

from model import evaluate
from misc import set_up_helper


class TestTaskOperators(unittest.TestCase):
    def setUp(self) -> None:
        pos_content = (
            'iou' + os.linesep +
            '' + os.linesep
        )
        set_up_helper('operators_pos_1', pos_content)
        pos_content = (
            'ptiou' + os.linesep +
            '' + os.linesep
        )
        set_up_helper('operators_pos_2', pos_content)
        pos_content = (
            'nms' + os.linesep +
            '' + os.linesep
        )
        set_up_helper('operators_pos_3', pos_content)
        pos_content = (
            'single_level_responsible_flags' + os.linesep +
            '' + os.linesep
        )
        set_up_helper('operators_pos_4', pos_content)
        pos_content = (
            'yolo_bbox_coder' + os.linesep +
            '' + os.linesep
        )
        set_up_helper('operators_pos_5', pos_content)
        pos_content = (
            'delta_xywh_bbox_coder' + os.linesep +
            '' + os.linesep
        )
        set_up_helper('operators_pos_6', pos_content)
        pos_content = (
            'channel_shuffle' + os.linesep +
            '' + os.linesep
        )
        set_up_helper('operators_pos_7', pos_content)
        pos_content = (
            'Prefetcher' + os.linesep +
            '' + os.linesep
        )
        set_up_helper('operators_pos_8', pos_content)
        pos_content = (
            'Dropout' + os.linesep +
            '' + os.linesep
        )
        set_up_helper('operators_pos_9', pos_content)
        pos_content = (
            'LabelSmoothingCrossEntropy' + os.linesep +
            '' + os.linesep
        )
        set_up_helper('operators_pos_10', pos_content)
        pos_content = (
            'ROIAlign' + os.linesep +
            '' + os.linesep
        )
        set_up_helper('operators_pos_11', pos_content)
        pos_content = (
            'DCNv2' + os.linesep +
            '' + os.linesep
        )
        set_up_helper('operators_pos_12', pos_content)
        pos_content = (
            'LSTM' + os.linesep +
            '' + os.linesep
        )
        set_up_helper('operators_pos_13', pos_content)
        neg_content = (
            '# iou' + os.linesep +
            '# ptiou' + os.linesep +
            '# nms' + os.linesep +
            '# single_level_responsible_flags' + os.linesep +
            '# yolo_bbox_coder' + os.linesep +
            '# delta_xywh_bbox_coder' + os.linesep +
            '# channel_shuffle' + os.linesep +
            '# Prefetcher' + os.linesep +
            '# Dropout' + os.linesep +
            '# LabelSmoothingCrossEntropy' + os.linesep +
            '# ROIAlign' + os.linesep +
            '# DCNv2' + os.linesep +
            '# LSTM' + os.linesep +
            '' + os.linesep
        )
        set_up_helper('operators_neg', neg_content)

    def tearDown(self) -> None:
        os.system('rm -rf ../data/project/test_task_operators_*')

    def test_task_operators_pos_1(self):
        res_s = evaluate('../data/project/test_task_operators_pos_1', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any('npu_iou' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_operators_pos_2(self):
        res_s = evaluate('../data/project/test_task_operators_pos_2', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any('npu_ptiou' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_operators_pos_3(self):
        res_s = evaluate('../data/project/test_task_operators_pos_3', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any(
            'npu_multiclass_nms' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_operators_pos_4(self):
        res_s = evaluate('../data/project/test_task_operators_pos_4', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any(
            'npu_single_level_responsible_flags' in item for r in res['extendResult'] for v in r['value'] for item in v
        ))

    def test_task_operators_pos_5(self):
        res_s = evaluate('../data/project/test_task_operators_pos_5', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any(
            'npu_bbox_coder_encode_yolo' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_operators_pos_6(self):
        res_s = evaluate('../data/project/test_task_operators_pos_6', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any(
            'npu_bbox_coder_encode_xyxy2xywh' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_operators_pos_7(self):
        res_s = evaluate('../data/project/test_task_operators_pos_7', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any('ChannelShuffle' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_operators_pos_8(self):
        res_s = evaluate('../data/project/test_task_operators_pos_8', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any('Prefetcher' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_operators_pos_9(self):
        res_s = evaluate('../data/project/test_task_operators_pos_9', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any('DropoutV2' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_operators_pos_10(self):
        res_s = evaluate('../data/project/test_task_operators_pos_10', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any(
            'LabelSmoothingCrossEntropy' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_operators_pos_11(self):
        res_s = evaluate('../data/project/test_task_operators_pos_11', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any('ROIAlign' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_operators_pos_12(self):
        res_s = evaluate('../data/project/test_task_operators_pos_12', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any('DCNv2' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_operators_pos_13(self):
        res_s = evaluate('../data/project/test_task_operators_pos_13', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '0')
        self.assertTrue(any('BiLSTM' in item for r in res['extendResult'] for v in r['value'] for item in v))

    def test_task_operators_neg(self):
        res_s = evaluate('../data/project/test_task_operators_neg', None)
        res = json.loads(res_s)
        self.assertEqual(res['classType'], '1')
        self.assertEqual(res['errorCode'], '1')


if __name__ == "__main__":
    unittest.main()
