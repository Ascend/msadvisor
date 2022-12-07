# -*- coding:utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/local')
import local
from utils import get_op_type
from parse import parse


def evaluate(data_path, parameters):
    root_node, lines = parse(data_path)
    if 'local' not in parameters:
        loc = 'en'
    elif parameters['local'] in ('zh', 'en'):
        loc = parameters['local']
    else:
        raise ValueError("parameter['local'] only support ['zh', 'en']")

    op_type = get_op_type(root_node)

    # Build
    local.set_locale(loc)
    from rules import Rule
    Rule.init_cls(lines)
    from node_visitor import DslOptimizer, TikOptimizer

    if op_type == 'DSL':
        visitor = DslOptimizer()
    elif op_type == 'TIK':
        visitor = TikOptimizer()
    else:
        raise ValueError('The file is not a DSL/TIK type operator file')
    visitor.visit(root_node)

    res = Rule.convert_results()
    return res.generate()


if __name__ == '__main__':
    dirname = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    test_data_path = dirname + '/test/test_ops_file/2_exp_neg_dsl.py'
    test_parameters = {'local': 'en'}
    evaluate(test_data_path, test_parameters)
