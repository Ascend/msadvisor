import os
import json

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory

class ExtendResult:
    def __init__(self):
        self.type = '0'
        self.extend_title = ""
        self.data_type = []      # table type is an array with multiple elements, list type with only one element
        self.key = []           # this field is only used for table type result
        self.value = []         # table type is a two-dimensional array, list type is a one-dimensional array

class Result:
    def __init__(self):
        self.class_type = '0'
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

def need_to_optimize(graph, knowledge):
    while knowledge.has_next_pattern():
        knowledge.next_pattern()
        match_results = knowledge.get_candidate_sub_graphs(graph)
        if match_results is None or len(match_results) == 0:
            continue
        else:
            return True
    return False

def optimize(graph, knowledge):
    res = True
    print("begin loop")
    while knowledge.has_next_pattern():
        print("next loop")
        knowledge.next_pattern()
        match_results = knowledge.get_candidate_sub_graphs(graph)
        if match_results is None or len(match_results) == 0:
            continue
        while knowledge.has_next_apply():
            knowledge.next_apply()
            for match_result in match_results:
                res &= knowledge.apply(graph, match_result)
    return res

def evaluate(datapath, parameter):
    # get parameter
    params = json.loads(parameter)
    mode = params.get("mode", "optimize")
    if mode not in [ "optimize", "evaluate" ]:
        raise RuntimeError('mode:{} invalid not in optimize or evaluate'.format(mode))

    file_name = params.get("file_name")
    if file_name is None:
        raise RuntimeError('file_name:{} is none'.format(file_name))
    onnx_path = os.path.join("{}/{}".format(datapath, file_name))
    if os.path.isfile(onnx_path) is False:
        raise RuntimeError('onnx_path:{} is not exist'.format(onnx_path))

    print("evaluate datapath:{} parameter:{} onnx_path:{}".format(datapath, parameter, onnx_path))

    # fill result
    result = Result()
    result.class_type = class_type['model']
    result.error_code = error_code['success']
    result.summary = "The current model is well optimized"

    # load model
    onnx_graph = OnnxGraph.parse(onnx_path)

    knowledge_name = list(KnowledgeFactory.get_knowledge_pool().keys())[0]

    knowlege_class = KnowledgeFactory.get_knowledge(knowledge_name)
    print("get knowledge name:{} class:{}".format(knowledge_name, knowlege_class))

    if mode == "evaluate":
        needflag =  need_to_optimize(onnx_graph, knowlege_class)
        if needflag is True:
            result.summary = "The current model need to be optimized"
    elif mode == "optimize":
        flag = optimize(onnx_graph, knowlege_class)
        if flag is True:
            out_file = os.path.join(datapath, "{}_optimize.onnx".format(os.path.splitext(file_name)[0]))
            onnx_graph.save(out_file)
            result.error_code = error_code['optimized']
            result.summary = "The current model need to be optimized,the optimized model path is:{}".format(out_file)
        else:
            raise RuntimeError('optimize failed file:{}'.format(onnx_path))

    return result.generate()