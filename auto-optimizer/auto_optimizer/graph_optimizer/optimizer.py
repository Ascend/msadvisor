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

from dataclasses import dataclass
from functools import partial
import os
import pathlib
import logging
import tempfile
from copy import deepcopy
from typing import Callable, Dict, List, Tuple

import numpy as np
from numpy.linalg import norm

from auto_optimizer.graph_refactor.interface.base_graph import BaseGraph
from auto_optimizer.pattern.knowledges.knowledge_base import KnowledgeBase
from auto_optimizer import KnowledgeFactory


logger = logging.getLogger('GraphOptimizer')


def close(mat0, mat1, threshold=1e-3) -> bool:
    mat0 = np.ndarray.flatten(mat0)
    mat1 = np.ndarray.flatten(mat1)
    cos_sim: float = np.dot(mat0, mat1) / (norm(mat0) * norm(mat1))
    return 1 - cos_sim < threshold


@dataclass
class InferTestConfig:
    '''Config class'''
    converter: str = 'atc'
    soc: str = 'Ascend310P3'
    device: int = 0
    loop: int = 100
    threshold: float = -0.02
    is_static: bool = True
    input_shape: str = ''
    input_shape_range: str = ''
    dynamic_shape: str = ''
    output_size: str = ''


class GraphOptimizer:
    '''Public Graph Optimizer class.'''
    def __init__(self, knowledges_: List[str]) -> None:
        registered_knowledges = KnowledgeFactory.get_knowledge_pool()
        for idx, name in enumerate(registered_knowledges):
            knowledges_ = [name if v == str(idx) else v for v in knowledges_]
        knowledges_ = list(dict.fromkeys(knowledges_))
        knowledge_dict = {
            name: registered_knowledges.get(name, KnowledgeBase())
            for name in knowledges_
            if name in registered_knowledges
        }
        if len(knowledge_dict) == 0:
            raise ValueError('No valid knowledge provided.')
        self.knowledges: Dict[str, KnowledgeBase] = knowledge_dict

    def load_config(self):
        pass

    @staticmethod
    def _effective(om_ori: str, om_opt: str, cfg: InferTestConfig) -> bool:
        from auto_optimizer.inference_engine.inference.acl_inference \
                import InferSession, tensor_type_to_numpy_type

        sess_ori = InferSession(device_id=cfg.device, model_path=om_ori, loop=cfg.loop)
        sess_opt = InferSession(device_id=cfg.device, model_path=om_opt, loop=cfg.loop)
        if cfg.is_static:
            input_ = [
                np.random.randn(*inp.shape)
                         .astype(tensor_type_to_numpy_type[inp.datatype])
                for inp in sess_ori.get_inputs()
            ]
            out_ori = sess_ori.infer(input_)
            out_opt = sess_opt.infer(input_)
        else:
            custom_sizes = int(cfg.output_size)
            dyn_shape = {
                k: [int(n) for n in v.split(',')]
                for k, v in [
                    inp.split(':') for inp in cfg.dynamic_shape.split(';')
                ]
            }
            input_ = [
                np.random.randn(*dyn_shape[inp.name])
                         .astype(tensor_type_to_numpy_type[inp.datatype])
                for inp in sess_ori.get_inputs()
            ]
            out_ori = sess_ori.infer(input_, mode='dymshape', custom_sizes=custom_sizes)
            out_opt = sess_opt.infer(input_, mode='dymshape', custom_sizes=custom_sizes)

        time_ori = np.mean(sess_ori.sumary().exec_time_list)
        time_opt = np.mean(sess_opt.sumary().exec_time_list)

        if out_ori is None or out_opt is None or len(out_ori) != len(out_opt):
            logger.warning('Optimization failed: result is wrong.')
            return False

        if not all(close(mat0, mat1) for mat0, mat1 in zip(out_ori, out_opt)):
            logger.warning('Optimization failed: result not close enough.')
            return False

        logger.info('Origin inference time: %.2f ms', time_ori)
        logger.info('Optimized inference time: %.2f ms', time_ori)
        logger.info('Inference speed improved %.2s%%.', (time_ori / time_opt - 1) * 100)
        if time_opt > time_ori * (1 - cfg.threshold):
            logger.warning('Optimization cancaled: not enough improvement.')
            return False
        return True

    @staticmethod
    def _evaluate(graph: BaseGraph, knowledge: KnowledgeBase) -> bool:
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
                    if knowledge.apply(graph, match_result):
                        return True
        return False

    @staticmethod
    def _optimize(graph: BaseGraph, knowledge: KnowledgeBase) -> bool:
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

    def _exec_action(
        self,
        graph: BaseGraph,
        action: Callable[[BaseGraph, KnowledgeBase], bool]
    ) -> Tuple[BaseGraph, List[str]]:
        applied_knowledges = []
        for name, knowledge in self.knowledges.items():
            knowledge.reset()
            graph_copy = deepcopy(graph)
            try:
                if action(graph_copy, knowledge):
                    graph = graph_copy
                    applied_knowledges.append(name)
            except Exception as exc:
                logger.warning('Error applying knowledge: %s!', name)
                logger.warning(exc)
        return graph, applied_knowledges

    def evaluate_knowledges(self, graph: BaseGraph) -> Tuple[BaseGraph, List[str]]:
        '''Test graph for applicable knowledges.'''
        return self._exec_action(graph, GraphOptimizer._evaluate)

    def apply_knowledges(self, graph: BaseGraph) -> Tuple[BaseGraph, List[str]]:
        '''Optimize graph using optimizer.'''
        return self._exec_action(graph, GraphOptimizer._optimize)

    def apply_knowledges_with_infer_test(
        self,
        graph: BaseGraph,
        cfg: InferTestConfig
    ) -> Tuple[BaseGraph, List[str]]:
        '''Optimize graph using optimizer, eliminate negative knowledges with
        inference testing.'''
        from auto_optimizer.inference_engine.model_convert import onnx2om
        from auto_optimizer.inference_engine.inference.acl_inference \
                import InferSession, tensor_type_to_numpy_type
        applied_knowledges = []
        tmp_dir = tempfile.gettempdir()
        pid = os.getpid()
        onnx_ori = pathlib.Path(tmp_dir, f'auto_optimizer_{pid}_ori.onnx')
        onnx_opt = pathlib.Path(tmp_dir, f'auto_optimizer_{pid}_opt.onnx')
        graph.save(onnx_ori.as_posix())
        cvtr_stc = partial(onnx2om, input_shape=cfg.input_shape) if cfg.input_shape else onnx2om
        cvtr_dyn = partial(onnx2om, input_shape_range=cfg.input_shape_range)
        cvtr = cvtr_stc if cfg.is_static else cvtr_dyn
        om_ori = cvtr(
            path_onnx=onnx_ori.as_posix(),
            converter=cfg.converter,
            soc_version=cfg.soc,
        )
        for name, knowledge in self.knowledges.items():
            logger.info('Evaluating knowledge %s', name)
            graph_opt = deepcopy(graph)
            knowledge.reset()
            try:
                if not self._optimize(graph_opt, knowledge):
                    continue
                graph_opt.save(onnx_opt.as_posix())
                om_opt = cvtr(
                    path_onnx=onnx_opt.as_posix(),
                    converter=cfg.converter,
                    soc_version=cfg.soc,
                )
                if self._effective(om_ori, om_opt, cfg=cfg):
                    applied_knowledges.append(name)
                    graph = graph_opt
                    os.rename(om_opt, om_ori)
            except Exception as exc:
                logger.warning("%s", exc)
        return graph, applied_knowledges


if __name__ == "__main__":
    knowledges = KnowledgeFactory.get_knowledge_pool()
    optimizer = GraphOptimizer(list(knowledges.keys()))
