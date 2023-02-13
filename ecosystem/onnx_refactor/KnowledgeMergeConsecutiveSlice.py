# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory


def evaluate(data_path, param):
    knowledge = KnowledgeFactory.get_knowledge('KnowledgeMergeConsecutiveSlice')
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    from advisor import evaluate_x
    return evaluate_x(knowledge, data_path, param)
