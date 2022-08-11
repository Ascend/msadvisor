# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import unittest

from auto_optimizer.common import Register
from auto_optimizer.inference_engine.data_process_factory import EvaluateFactory
from auto_optimizer.inference_engine.data_process_factory import PreProcessFactory
from auto_optimizer.inference_engine.data_process_factory import PostProcessFactory


class TestResnet50(unittest.TestCase):

    def setUp(self) -> None:
        sys.path.append("..")
        os.chdir("..")

        register = Register(os.path.join(os.getcwd(), "auto_optimizer", "inference_engine"))
        register.import_modules()

    def tearDown(self) -> None:
        pass

    def test_resnet50_pre_process(self):
        for name, pre_process in PreProcessFactory.get_pre_process_pool().items():
            dataset = {"src" : "/home/", "dest" : "/home/", "label" : "/home/"}
            ret = pre_process(dict=dataset)
            self.assertEqual(True, ret)

    def test_resnet50_pre_process_by_name(self):
        pre_process = PreProcessFactory.get_pre_process("ImageNet")
        if pre_process != "Not exist":
            ret = pre_process(str="hello", len=10, path="/home")
            self.assertEqual(True, ret)

    def test_resnet50_pre_process_by_error_name(self):
        pre_process = PreProcessFactory.get_pre_process("Imagenet")
        self.assertEqual("Not exist", pre_process)

    def test_resnet50_post_process(self):
        for name, post_process in PostProcessFactory.get_post_process_pool().items():
            ret = post_process(str="hello", len=10, path="/home")
            self.assertEqual(True, ret)

    def test_resnet50_evaluate(self):
        for name, evaluate in EvaluateFactory.get_evaluate_pool().items():
            ret = evaluate(str="hello", len=10, path="/home")
            self.assertEqual(True, ret)

            ret = evaluate.acc_1(str="hello", len=10, path="/home")
            self.assertEqual(True, ret)


def test_suite():
    suite = unittest.TestSuite()

    suite.addTest(TestResnet50("test_resnet50_pre_process"))
    suite.addTest(TestResnet50("test_resnet50_pre_process_by_name"))
    suite.addTest(TestResnet50("test_resnet50_pre_process_by_error_name"))
    suite.addTest(TestResnet50("test_resnet50_post_process"))

    return suite
