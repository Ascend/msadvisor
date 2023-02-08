# -*- coding:utf-8 -*-
import ast


def find_var_domain(node):
    while not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        node = node.parent
    last_node = node
    while hasattr(last_node, 'body'):
        last_node = last_node.body[-1]
    node.endno = last_node.lineno
    return node


def parse(data_path):
    with open(data_path, encoding='utf-8') as f:
        lines = f.readlines()
        code = ''.join(lines)
    root_node = ast.parse(code)
    # 构造AST子节点到父节点的引用
    for node in ast.walk(root_node):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    return root_node, lines


