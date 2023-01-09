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
import json
import stat
import sys
import fcntl
import shutil

import msadvisor as ms

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.graph_optimizer import GraphOptimizer, InferTestConfig
from auto_optimizer.pattern import KnowledgeBase, KnowledgeFactory

LOG_INFO = 1
LOG_WARN = 2
LOG_ERR = 3


class ExtendResult:
    def __init__(self):
        self.type = '0'
        self.extend_title = ""
        self.data_type = []     # table type is an array with multiple elements, list type with only one element
        self.key = []           # this field is only used for table type result
        self.value = []         # table type is a two-dimensional array, list type is a one-dimensional array


class Result:
    def __init__(self):
        self.class_type = '1'
        self.error_code = '0'
        self.summary = ""
        self.extend_result = []

    def generate(self):
        extend_data = []
        for item in self.extend_result:
            data = {"type": item.type, "extendTitle": item.extend_title,
                    "dataType": item.data_type, "key": item.key, "value": item.value}
            extend_data.append(data)
        res = {"classType": self.class_type, "errorCode": self.error_code,
               "summary": self.summary, "extendResult": extend_data}
        outputstr = json.dumps(res)
        return outputstr

class_type = {'op': '0', 'model': '1'}
error_code = {'success': '0', 'optimized': '1'}
extend_type = {'list': '0', 'table': '1', 'sourcedata': '2'}
extend_data_type = {'str': '0', 'int': '1', 'double': '2'}

def is_graph_input_static(graph: OnnxGraph) -> bool:
    for inp in graph.inputs:
        for dim in inp.shape:
            try:
                dim = int(dim)
                if dim <= 0:
                    return False
            except ValueError:
                return False
    return True

def get_default_soc():
    try:
        import acl
        return acl.get_soc_name()
    except Exception:
        return 'Ascend310P3'

def check_file_permission(filepath):
    if not os.path.isfile(filepath):
        ms.utils.log(LOG_WARN, 'file not exist, check permission failed, file={}.'.format(filepath))
        return False
    filestat = os.stat(filepath)
    # check file owner
    if filestat.st_uid != os.getuid() or filestat.st_gid != os.getgid():
        ms.utils.log(LOG_WARN, 'file owner not equal current execute user.')
        return False
    # check file permission
    if filestat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        ms.utils.log(LOG_WARN, 'file permission support group or other user to write.')
        return False
    return True

def find_one_model_file(path, suffix = '.onnx'):
    if not os.path.exists(path):
        return None
    files = os.listdir(path)
    for filename in files:
        if filename.endswith(suffix):
            return filename
    return None

def lock_file(handle):
    if handle is None:
        ms.utils.log(LOG_ERR, 'file handle is invalid.')
        return False
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
    except OSError as err:
        ms.utils.log(LOG_ERR, 'lock file failed, err: %s' % err)
        return False
    return True

def convert_to_str(match_result):
    return ','.join(
        '(' + ','.join(node.name for nodes in node_dict.values() for node in nodes) + ')'
        for node_dict in match_result.node_dicts
    )

def analyze_model(graph: OnnxGraph, knowledge: KnowledgeBase, result: Result):
    extend_result = ExtendResult()
    while knowledge.has_next_pattern():
        knowledge.next_pattern()
        match_results = knowledge.match_pattern(graph)
        if match_results is None:
            continue
        for match_result in match_results:
            result_str = convert_to_str(match_result)
            extend_result.value.append([result_str])
    if len(extend_result.value) != 0:
        result.summary = "Exist optimized nodes, please view details."
        extend_result.type = extend_type['table']
        extend_result.extend_title = 'Optimized subgraph'
        extend_result.key.append('Optimized subgraph nodes')
        extend_result.data_type.append(extend_data_type['str'])
        result.extend_result.append(extend_result)

def optimize_model(graph: OnnxGraph, knowledge: KnowledgeBase, result: Result, out_path):
    optimize_result = False
    while knowledge.has_next_pattern():
        knowledge.next_pattern()
        match_results = knowledge.match_pattern(graph)
        if match_results is None or len(match_results) == 0:
            continue
        while knowledge.has_next_apply():
            knowledge.next_apply()
            for match_result in match_results:
                # if apply() return True, then graph must be changed.
                optimize_result |= knowledge.apply(graph, match_result)
    if optimize_result:
        # graph is optimized.
        result.error_code = error_code['optimized']
        result.summary = "The current model has already been optimized, the optimized model path is:%s" % out_path
    return optimize_result

def evaluate_x(knowledge: KnowledgeBase, datapath, parameter):
    """
    inferface for the msadvisor command
    :param knowledge: model refactor knowledge object
    :param datapath: model path
    :param parameter: knowledge input parameters
    :return: optimization result, if exist error, will raise runtime error exception
    """
    # get parameter
    params = json.loads(parameter)
    mode = params.get("mode", "optimize")
    if mode not in [ "optimize", "evaluate" ]:
        raise RuntimeError('mode:{} invalid not in optimize or evaluate'.format(mode))

    model_file = params.get("model_file")
    datapath = os.path.realpath(datapath)
    sub_path = 'project' # adapter for IDE

    onnx_path = None
    if model_file is None or model_file == '':
        # find onnx model in datapath
        model_file = find_one_model_file(os.path.join(datapath, sub_path)) # adapter for IDE
        if model_file is None:
            ms.utils.log(LOG_WARN, 'There is no model file in {datapath}/project')
            model_file = find_one_model_file(datapath)
            if model_file is None:
                raise RuntimeError('model file not exist in datapath')
            onnx_path = os.path.join(datapath, model_file)
            sub_path = ''
        else:
            onnx_path = os.path.join(os.path.join(datapath, sub_path), model_file)
    else:
        # check onnx model do or not exist
        onnx_path = os.path.realpath(os.path.join(os.path.join(datapath, sub_path), model_file))
        if not os.path.isfile(onnx_path):
            onnx_path = os.path.realpath(os.path.join(datapath, model_file))
            if not os.path.isfile(onnx_path):
                raise RuntimeError('model file not exist, filename={}'.format(model_file))
            sub_path = ''

    # fill result
    result = Result()
    result.class_type = class_type['model']
    result.error_code = error_code['success']
    result.summary = "The current model is well optimized."

    # evaluate onnx model
    if mode == 'evaluate':
        # load source model
        onnx_graph = OnnxGraph.parse(onnx_path)
        if len(onnx_graph.inputs) == 0 and len(onnx_graph.outputs) == 0:
            raise RuntimeError('The current model is invalid.')
        analyze_model(onnx_graph, knowledge, result)
        return result.generate()

    # optimize onnx model
    # synchronize multi process apply knowledge
    if not lock_file(open(onnx_path)):
        raise RuntimeError('Lock onnx file failed.')
    # get optimized onnx output path
    out_path = os.path.join(os.path.join(datapath, sub_path), 'out')
    if not (os.path.exists(out_path) and os.path.isdir(out_path)):
        if os.path.exists(out_path) and not os.path.isdir(out_path):
            os.remove(out_path)
        os.mkdir(out_path)
        os.chmod(out_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    new_onnx_path = os.path.join(out_path, '%s_%d_optimize.onnx' % (os.path.splitext(model_file)[0], os.getpid()))
    if os.path.isfile(new_onnx_path):
        # load model which other knowledge has bean optimized, and continue with the previous optimization
        if not check_file_permission(new_onnx_path):
            raise RuntimeError('The output onnx path has bean existed, but no permission.')
        if not lock_file(open(new_onnx_path)):
            raise RuntimeError('Lock new onnx file failed.')
        onnx_graph = OnnxGraph.parse(new_onnx_path)
    else:
        # load source model
        onnx_graph = OnnxGraph.parse(onnx_path)
    is_onnx_static = is_graph_input_static(graph=onnx_graph)
    if len(onnx_graph.inputs) == 0 and len(onnx_graph.outputs) == 0:
        raise RuntimeError('The current model is invalid.')
    knowledge_name = knowledge.__class__.__name__
    pool = KnowledgeFactory.get_knowledge_pool()
    for name, know in pool.items():
        if know == knowledge:
            knowledge_name = name
    infer_test = params.get('infer_test', False)
    optimizer = GraphOptimizer([knowledge_name])
    if infer_test:
        attrs = {
            'converter': 'atc',
            'soc': get_default_soc(),
            'device': 0,
            'loop': 100,
            'threshold': 0,
            'is_static': is_onnx_static,
            'input_shape': '',
            'input_shape_range': '',
            'dynamic_shape': '',
            'output_size': '',
            'process_run_infer': True,
        }
        for para in attrs:
            if para in params:
                attrs[para] = params[para]
        cfg = InferTestConfig(**attrs)
        if not (cfg.is_static or (cfg.input_shape_range and cfg.dynamic_shape and cfg.output_size)):
            raise RuntimeError(
                'Didn\'t specify input_shape_range or dynamic_shape or output_size'
                ' for dynamic input shape onnx in inference test'
            )
        onnx_graph, applied_knowledges = optimizer.apply_knowledges_with_infer_test(onnx_graph, cfg)
    else:
        onnx_graph, applied_knowledges = optimizer.apply_knowledges(onnx_graph)
    if applied_knowledges:
        result.error_code = error_code['optimized']
        result.summary = "The current model has already been optimized, the optimized model path is:%s" % out_path
    if applied_knowledges:
        if params.get('extract'):
            input_names = [i.name for i in onnx_graph.inputs]
            output_names = [i.name for i in onnx_graph.outputs]
            onnx_graph.extract(new_onnx_path, input_names, output_names)
        else:
            onnx_graph.save(new_onnx_path)
        # replace result for IDE
        if sub_path == 'project': # adapter for IDE
            ide_out_path = os.path.join(datapath, '%s_optimize.onnx' % os.path.splitext(model_file)[0])
            shutil.copy(new_onnx_path, ide_out_path)
            result.summary = "The current model has already been optimized, \
the optimized model path is:%s" % ide_out_path
    return result.generate()
