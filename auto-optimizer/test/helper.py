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


from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from numpy.typing import NDArray
import onnxruntime as ort

from auto_optimizer.common.utils import meet_precision
from auto_optimizer.graph_optimizer.optimizer import GraphOptimizer
from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase

ort.set_default_logger_severity(3)


@dataclass
class OptimizationConfig:
    graph: BaseGraph
    knowledge: KnowledgeBase
    onnx_ori: str = ''
    onnx_opt: str = ''


class KnowledgeTestHelper:
    @staticmethod
    def graph_equal(lhs: BaseGraph, rhs: BaseGraph) -> bool:
        '''检查两个图是否等价，检查图的inputs/outputs/initializers/nodes的相对关系是否相同。
        比如如果只是做infershape，则认为前后的图仍是同一个。'''
        if not (isinstance(rhs, BaseGraph) and isinstance(lhs, BaseGraph)):
            return False
        if lhs.name != rhs.name:
            return False
        inputs_lhs = {inp.name: inp for inp in lhs.inputs}
        inputs_rhs = {inp.name: inp for inp in rhs.inputs}
        if inputs_lhs != inputs_rhs:
            return False
        outputs_lhs = {out.name: out for out in lhs.outputs}
        outputs_rhs = {out.name: out for out in rhs.outputs}
        if outputs_lhs != outputs_rhs:
            return False
        inits_lhs = {ini.name: ini for ini in lhs.initializers}
        inits_rhs = {ini.name: ini for ini in rhs.initializers}
        if inits_lhs != inits_rhs:
            return False
        inmap_lhs, inmap_rhs = defaultdict(dict), defaultdict(dict)
        for node in lhs.nodes:
            for inp in node.inputs:
                inmap_lhs[inp][node.name] = node
        for node in rhs.nodes:
            for inp in node.inputs:
                inmap_rhs[inp][node.name] = node
        return inmap_lhs == inmap_rhs

    def inference(self, onnx_path: str, feeds: List[Dict[str, NDArray]]) -> List[List[NDArray]]:
        '''Inference a onnx model with a list of feeds'''
        session = ort.InferenceSession(onnx_path)
        outputs_name = [meta.name for meta in session.get_outputs()]
        return [session.run(outputs_name, feed) for feed in feeds]

    def optimize(self, graph: BaseGraph, knowledge: KnowledgeBase) -> Tuple[bool, BaseGraph]:
        '''Optimize a graph with specific knowledge.'''
        graph_opt = deepcopy(graph)
        res = GraphOptimizer._optimize(graph_opt, knowledge)
        return res, graph_opt

    def _check_optimization_failure(self, cfg: OptimizationConfig) -> bool:
        success, graph_opt = self.optimize(cfg.graph, cfg.knowledge)
        return not success and self.graph_equal(cfg.graph, graph_opt)

    def check_optimization(self, cfg: OptimizationConfig, expect: bool) -> bool:
        '''Perferm optimization with the provided config, check if the result is as expected.'''
        return self._check_optimization_success(cfg) if expect \
            else self._check_optimization_failure(cfg)

    def _check_optimization_success(self, cfg: OptimizationConfig) -> bool:
        success, graph_opt = self.optimize(cfg.graph, cfg.knowledge)
        if not success or self.graph_equal(cfg.graph, graph_opt):
            return False
        success, graph_opt_2 = self.optimize(graph_opt, cfg.knowledge)
        if success or not self.graph_equal(graph_opt, graph_opt_2):
            return False
        cfg.graph.save(cfg.onnx_ori)
        graph_opt.save(cfg.onnx_opt)
        return True

    def check_precision(
        self,
        onnx_ori: str,
        onnx_opt: str,
        feeds: List[Dict[str, NDArray[Any]]],
        cos_th: float = 1e-6,
        atol: float = 1e-8,
        rtol: float = 1e-5
    ) -> bool:
        '''Check inference precision of two graph.'''
        outs_ori = self.inference(onnx_ori, feeds)
        outs_opt = self.inference(onnx_opt, feeds)
        for out_ori, out_opt in zip(outs_ori, outs_opt):
            if len(out_ori) != len(out_opt):
                return False
            if not all(
                meet_precision(lmat, rmat, cos_th=cos_th, rtol=rtol, atol=atol)
                for lmat, rmat in zip(out_ori, out_opt)
            ):
                return False
        return True
