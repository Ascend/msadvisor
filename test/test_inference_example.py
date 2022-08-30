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
import unittest
from multiprocessing import Pool, Manager

import timeout_decorator

from auto_optimizer.common import Register
from auto_optimizer.common.config import Config

from auto_optimizer.inference_engine.data_process_factory import InferenceFactory
from auto_optimizer.inference_engine.data_process_factory import EvaluateFactory
from auto_optimizer.inference_engine.data_process_factory import PreProcessFactory
from auto_optimizer.inference_engine.data_process_factory import PostProcessFactory


class TestResnet(unittest.TestCase):

    def setUp(self) -> None:

        register = Register(os.path.join(os.getcwd(), "auto_optimizer"))
        register.import_modules()

        self.cfg = Config.read_by_file(os.sep.join(['auto_optimizer', 'configs', 'cv',
                                                    'classification', 'example.py']))

        max_queue_num = 100
        self.pre_queue = Manager().Queue(max_queue_num)
        self.infer_queue = Manager().Queue(max_queue_num)
        self.post_queue = Manager().Queue(max_queue_num)

        self.pre_process_pool = None
        self.post_process_pool = None
        self.inference_pool = None
        self.evaluate_pool = None

    def tearDown(self) -> None:
        pass

    @timeout_decorator.timeout(seconds=10)
    def inference(self, cfg):
        try:
            batch_size = cfg["batch_size"]
            engine_cfg = cfg["engine"]

            worker = engine_cfg["pre_process"]["worker"]
            dataset_path = engine_cfg["pre_process"]["dataset_path"]
            real_path = os.path.realpath(dataset_path)
            file_len = len(os.listdir(real_path))

            # 计算inference等进程循环次数，数据考虑对齐
            remainder = file_len % (worker * batch_size)
            if remainder:
                loop = (file_len + (worker * batch_size - remainder)) // batch_size
            else:
                loop = file_len // batch_size
            print("loop={}".format(loop))

            self._thread(loop, worker, batch_size, engine_cfg)
        except Exception as err:
            raise RuntimeError("inference failed error={}".format(err))

    def _thread(self, loop, worker, batch_size, engine_cfg):
        pre_process, post_process, inference, evaluate = \
            self._get_engine(engine_cfg)

        self.pre_process_pool = Pool(worker)
        self.post_process_pool = Pool(1)
        self.inference_pool = Pool(1)
        self.evaluate_pool = Pool(1)

        for i in range(worker):
            self.pre_process_pool.apply_async(pre_process,
                                              args=(i, batch_size, worker, engine_cfg["pre_process"],
                                                    None, self.pre_queue))

        # 用单进程
        self.inference_pool.apply_async(inference,
                                        args=(loop, engine_cfg["inference"], self.pre_queue, self.infer_queue))
        self.post_process_pool.apply_async(post_process,
                                           args=(loop, engine_cfg["post_process"], self.infer_queue, self.post_queue))
        self.evaluate_pool.apply_async(evaluate,
                                       args=(loop, batch_size, engine_cfg["evaluate"], self.post_queue, None))

        self.pre_process_pool.close()
        self.inference_pool.close()
        self.post_process_pool.close()
        self.evaluate_pool.close()

        self.pre_process_pool.join()
        self.inference_pool.join()
        self.post_process_pool.join()
        self.evaluate_pool.join()

    def _get_engine(self, engine):
        try:
            pre_process = PreProcessFactory.get_pre_process(engine["pre_process"]["type"])
            post_process = PostProcessFactory.get_post_process(engine["post_process"]["type"])
            inference = InferenceFactory.get_inference(engine["inference"]["type"])
            evaluate = EvaluateFactory.get_evaluate(engine["evaluate"]["type"])
        except Exception as err:
            raise RuntimeError("get params failed error={}".format(err))

        return pre_process, post_process, inference, evaluate

    def test_resnet_inference(self):
        self.inference(self.cfg)


def test_suite():
    suite = unittest.TestSuite()

    suite.addTest(TestResnet("test_resnet_inference"))

    return suite
