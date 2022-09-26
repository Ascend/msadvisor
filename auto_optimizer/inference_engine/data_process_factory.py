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


from typing import Dict, Type
from .pre_process.pre_process_base import PreProcessBase
from .post_process.post_process_base import PostProcessBase
from .evaluate.evaluate_base import EvaluateBase
from .inference.inference_base import InferenceBase
from .datasets.dataset_base import DatasetBase

from ..common.utils import typeassert


class EngineFactoryBase(object):
    _engine_pool = {}

    @classmethod
    def add_engine(cls, name, engine):
        cls._engine_pool[name] = engine

    @classmethod
    @typeassert(name=str)
    def get_engine(cls, name):
        return cls._engine_pool.get(name, None)

    @classmethod
    @typeassert(name=str)
    def register(cls, name):
        def _wrapper(dataset_cls):
            cls.add_engine(name, dataset_cls())
            return dataset_cls
        return _wrapper


class DatasetFactory(object):
    _dataset_pool: Dict[str, DatasetBase] = {}

    @classmethod
    @typeassert(name=str, dataset=DatasetBase)
    def add_dataset(cls, name, dataset):
        cls._dataset_pool[name] = dataset

    @classmethod
    @typeassert(name=str)
    def get_dataset(cls, name):
        return cls._dataset_pool.get(name, None)

    @classmethod
    @typeassert(name=str)
    def register(cls, name):
        def _wrapper(dataset_cls: Type[DatasetBase]):
            cls.add_dataset(name, dataset_cls())
            return dataset_cls
        return _wrapper


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

    @classmethod
    @typeassert(name=str)
    def register(cls, name):
        def _wrapper(preprocess_cls: Type[PreProcessBase]):
            cls.add_pre_process(name, preprocess_cls())
            return preprocess_cls
        return _wrapper


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

    @classmethod
    @typeassert(name=str)
    def register(cls, name):
        def _wrapper(inference_cls: Type[InferenceBase]):
            cls.add_inference(name, inference_cls())
            return inference_cls
        return _wrapper


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

    @classmethod
    @typeassert(name=str)
    def register(cls, name):
        def _wrapper(post_process_cls: Type[PostProcessBase]):
            cls.add_post_process(name, post_process_cls())
            return post_process_cls
        return _wrapper


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

    @classmethod
    @typeassert(name=str)
    def register(cls, name):
        def _wrapper(evaluate_cls: Type[EvaluateBase]):
            cls.add_evaluate(name, evaluate_cls())
            return evaluate_cls
        return _wrapper
