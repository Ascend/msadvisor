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


from typing import Sequence

import onnxruntime as ort

from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph

ort.set_default_logger_severity(3)


def inference(onnx_path, x):
    session = ort.InferenceSession(onnx_path)
    inputs = session.get_inputs()
    outputs_name = [meta.name for meta in session.get_outputs()]
    if not isinstance(x, Sequence):
        x = [x]
    assert len(inputs) == len(x)
    feed = {inp.name: data for inp, data in zip(inputs, x)}
    return session.run(outputs_name, feed)


def optimize(graph: BaseGraph, knowledge: KnowledgeBase):
    res = False
    if not knowledge.pre_process(graph):
        return False
    while knowledge.has_next_pattern():
        knowledge.next_pattern()
        match_results = knowledge.match_pattern(graph)
        if match_results is None or len(match_results) == 0:
            continue
        while knowledge.has_next_apply():
            knowledge.next_apply()
            for match_result in match_results:
                res |= knowledge.apply(graph, match_result)
    return knowledge.post_process(graph) and res
