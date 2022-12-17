import ast
import tbe
from rules import get_dsl_rules_in_call, get_dsl_rules_in_def, get_tik_rules_in_call


class DslOptimizer(ast.NodeVisitor):
    def visit_Call(self, node):
        rules = get_dsl_rules_in_call()
        for detector in rules:
            detector(node)
        self.generic_visit(node)  # 递归遍历子节点

    def visit_FunctionDef(self, node):
        rules = get_dsl_rules_in_def()
        for detector in rules:
            detector(node)
        self.generic_visit(node)  # 递归遍历子节点


class TikOptimizer(ast.NodeVisitor):
    def __init__(self):
        self.soc = {
            'cores': tbe.common.platform.get_soc_spec("CORE_NUM")
        }

    def visit_Call(self, node: ast.Call):
        rules = get_tik_rules_in_call()
        for detector in rules:
            detector(node, self.soc)
        self.generic_visit(node)  # 递归遍历子节点
