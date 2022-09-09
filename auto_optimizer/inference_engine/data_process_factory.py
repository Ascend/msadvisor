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


from typing import Dict
from .pre_process.pre_process_base import PreProcessBase
from .post_process.post_process_base import PostProcessBase
from .evaluate.evaluate_base import EvaluateBase
from .inference.inference_base import InferenceBase

from ..common.utils import typeassert


class PreProcessFactory(object):
    _pre_process_pool: Dict[str, PreProcessBase] = {}

    @classmethod
    @typeassert(name=str, pre_process=PreProcessBase)
    def add_pre_process(cls, name, pre_process):
        cls._pre_process_pool[name] = pre_process

    @classmethod
    @typeassert(name=str)
    def get_pre_process(cls, name):
        return cls._pre_process_pool.get(name, None)


class InferenceFactory(object):
    _inference_pool: Dict[str, InferenceBase] = {}

    @classmethod
    @typeassert(name=str, inference=InferenceBase)
    def add_inference(cls, name, inference):
        cls._inference_pool[name] = inference

    @classmethod
    @typeassert(name=str)
    def get_inference(cls, name):
        return cls._inference_pool.get(name, None)


class PostProcessFactory(object):
    _post_process_pool: Dict[str, PostProcessBase] = {}

    @classmethod
    @typeassert(name=str, post_process=PostProcessBase)
    def add_post_process(cls, name, post_process):
        cls._post_process_pool[name] = post_process

    @classmethod
    @typeassert(name=str)
    def get_post_process(cls, name):
        return cls._post_process_pool.get(name, None)


class EvaluateFactory(object):
    _evaluate_pool: Dict[str, EvaluateBase] = {}

    @classmethod
    @typeassert(name=str, evaluate=EvaluateBase)
    def add_evaluate(cls, name, evaluate):
        cls._evaluate_pool[name] = evaluate

    @classmethod
    @typeassert(name=str)
    def get_evaluate(cls, name):
        return cls._evaluate_pool.get(name, None)

