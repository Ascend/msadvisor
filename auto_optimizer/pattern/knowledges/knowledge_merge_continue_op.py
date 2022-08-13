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
import json
import numpy as np
from knowledge_base import KnowledgeBase

def get_continuous_op(graph, op_type='Slice'):
    """
    @des        get continuous same type nodes
    @param      graph: input onnx graph
                op_type: the same op type
    @return     continuous op list
    """
    all_ops = graph.get_nodes(op_type)
    res = []
    # init mark list, -1 means not marked
    flags = [-1] * len(all_ops)
    for idx, node in enumerate(all_ops):
        # TODO: multi outputs scenes not inclued here
        pre_node = graph[node.inputs[0]]
        # print("loop get idx:{} node:{} prenode:{}".format(idx, node, pre_node))
        if pre_node in all_ops:
            pre_idx = all_ops.index(pre_node)
            if flags[idx] == -1 and flags[pre_idx] == -1:
                # both two nodes are not in list: add new node sequence
                res.append([node, pre_node])
                flags[idx] =  flags[pre_idx] = len(res) - 1
            elif flags[idx] != -1 and flags[pre_idx] == -1:
                # only cur_node in list: append pre node in list
                res[flags[idx]].append(pre_node)
                flags[pre_idx] = flags[idx]
            elif flags[idx] == -1 and flags[pre_idx] != -1:
                # only pre node in list: insert cur node in list
                res_idx = res[flags[pre_idx]].index(pre_node)
                res[flags[pre_idx]].insert(res_idx, node)
                flags[idx] = flags[pre_idx]
            else:
                # both pre node and cur node in list: concat two sequence
                res[flags[idx]] = res[flags[idx]] + res[flags[pre_idx]]
                flags[pre_idx] = flags[idx]
    flags = list(filter(lambda x: x != -1, flags))
    uniq_flags = []
    for f in flags:
        if f not in uniq_flags:
            uniq_flags.append(f)
    # inverted front and back
    return [res[idx][::-1] for idx in uniq_flags]

class KnowledgeMergeContinueOp(KnowledgeBase):
    def __init__(self):
        self.pattern_list = []

    def merge_intializers(self, graph, initializer1, initializer2, merged_name):
        """
        @des        merge two initializers to one initializer
        @param      graph: input onnx graph
                    initializer1: initializer need to be merged
                    initializer2: initializer need to be merged
                    merged_name: name for merged node
        @return     merged initializer
        """
        # print("lcm debug i1value:{} type:{}".format(initializer1.value, type(initializer1.value)))
        merged_data = np.append(
            initializer1.value,
            initializer2.value,
        )

        merged_node = graph.add_initializer(name=merged_name, value=merged_data)
        return merged_node

    def merge_slicedop(self, graph, node1, node2):
        """
        @des        merge two node to one node
        @param      graph: input onnx graph
                    slice_node1: slice node1 need to be merged
                    slice_node2: slice node2 need to be merged
        @return     merged graph
        """
        if node1.name == node2.name:
            return graph
        
        # modify slice_node1 -> merge_node
        node2.inputs[1] = self.merge_intializers(
            graph,
            graph[node1.inputs[1]],
            graph[node2.inputs[1]],
            '{}_1'.format(node1.name)).name
        node2.inputs[2] = self.merge_intializers(
            graph,
            graph[node1.inputs[2]],
            graph[node2.inputs[2]],
            '{}_2'.format(node1.name)).name
        node2.inputs[3] = self.merge_intializers(
            graph,
            graph[node1.inputs[3]],
            graph[node2.inputs[3]],
            '{}_3'.format(node1.name)).name
        node2.inputs[4] = self.merge_intializers(
            graph,
            graph[node1.inputs[4]],
            graph[node2.inputs[4]],
            '{}_4'.format(node1.name)).name
        graph.del_node(node1.name)
        return graph

    def pattern(self):
        print("KnowledgeExample pattern")
        pass

    def need_to_optimize(self, graph) -> bool:
        match_result = get_continuous_op(graph)
        return True if len(match_result) > 0 else False

    def optimize(self, graph) -> bool:
        print("optimize start")
        continuous_slice_nodes = get_continuous_op(graph)
        flag = False
        for nodes in continuous_slice_nodes:
            print("now merge node:{}".format(nodes))
            for node in nodes:
                graph = self.merge_slicedop(graph, node, nodes[-1])
            flag = True
        print("optimize end ret:", flag)
        return graph, flag

def evaluate(datapath, parameter):    
    # get parameter
    params = json.loads(parameter)
    file_name = params.get("file_name", "origin.onnx")
    action = params.get("action", "optimize")
    onnx_path = os.path.join("{}/{}".format(datapath, file_name))
    print("evaluate datapath:{} parameter:{} onnx_path:{}".format(datapath, parameter, onnx_path))

    # fill result
    import msadvisor_adapter
    result = msadvisor_adapter.Result()
    result.class_type = msadvisor_adapter.class_type['model']
    result.error_code = msadvisor_adapter.error_code['success']
    result.summary = "model no need to optimize"

    # load model
    from magiconnx.graph import OnnxGraph
    onnx_graph = OnnxGraph(onnx_path)

    knowledge = KnowledgeMergeContinueOp()
    needflag =  knowledge.need_to_optimize(onnx_graph)

    if needflag == True:
        if action == "evaluate":
            result.summary = "model have optimize graph need to optimize"
        elif action == "optimize":
            optimiszer_graph, flag = knowledge.optimize(onnx_graph)
            if flag == True:
                out_file = os.path.join(datapath, "{}_optimize.onnx".format(os.path.splitext(file_name)[0]))
                optimiszer_graph.save(out_file)
                result.error_code = msadvisor_adapter.error_code['optimized']
                result.summary = "model have optimize graph,optimize OK result file:{}".format(out_file)
            else:
                result.summary = "model have optimize graph,optimize failed"
    return result.generate()

if __name__ == "__main__":
    data_path, filename = os.path.split(sys.argv[1])
    action = "optimize" if len(sys.argv) < 3 else sys.argv[2]
    parameter= json.dumps({ "file_name" : filename, "action":action })
    print("evaluate start data_path:{} parameter:{}".format(data_path, parameter))
    ret = evaluate(data_path, parameter)
    print("evaluate called ret:\n{}\n".format(ret))
