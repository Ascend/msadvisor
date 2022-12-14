# -*- coding:utf-8 -*-
import ast


def get_op_type(root):
    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.Import):
            name = node.names[0].name
            if name == 'tbe.tik':
                return 'TIK'
            elif name == 'tbe.dsl':
                return 'DSL'
        elif isinstance(node, ast.ImportFrom):
            if node.module == 'tbe':
                name = node.names[0].name
                if name == 'tik':
                    return 'TIK'
                elif name == 'dsl':
                    return 'DSL'
