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
from abc import ABC

from ..evaluate_base import EvaluateBase
from ...data_process_factory import EvaluateFactory


class ClassificationEvaluate(EvaluateBase, ABC):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        print("evaluate")
        return True

    def acc_1(self, *args, **kwargs):
        print("acc_1")
        return True

    def acc_k(self, *args, **kwargs):
        print("evaluate")
        pass

EvaluateFactory.add_evaluate("Classification", ClassificationEvaluate())