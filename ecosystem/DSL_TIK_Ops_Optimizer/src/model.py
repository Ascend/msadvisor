# -*- coding:utf-8 -*-
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/local')
import local
from utils import get_op_type
from parse import parse


def evaluate_helper(file_path):
    root_node, lines = parse(file_path)
    op_type = get_op_type(root_node)

    # Build
    from rules import Rule
    Rule.init_cls(file_path, lines)
    from node_visitor import DslOptimizer, TikOptimizer

    if op_type == 'DSL':
        visitor = DslOptimizer()
    elif op_type == 'TIK':
        visitor = TikOptimizer()
    else:
        raise ValueError('The file is not a DSL/TIK type operator file')
    visitor.visit(root_node)


def evaluate(data_path, parameters):
    if 'local' not in parameters:
        loc = 'en'
    elif parameters['local'] in ('zh', 'en'):
        loc = parameters['local']
    else:
        raise ValueError("parameter['local'] only support ['zh', 'en']")

    local.set_locale(loc)

    from rules import Rule
    for root, dirs, files in os.walk(data_path):
        for file in files:
            full_path = os.path.join(root, file)
            evaluate_helper(full_path)
    res = Rule.convert_results()
    return res.generate()


if __name__ == '__main__':
    dirname = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    test_data_path = dirname + '/test/test_ops_file'
    test_parameters = {'local': 'en'}
    res_str = evaluate(test_data_path, test_parameters)
    res_json = json.loads(res_str)
    res_pretty = json.dumps(res_json, indent=4, ensure_ascii=False)
    print(res_pretty)
