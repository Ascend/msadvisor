# -*- coding:utf-8 -*-
import ast

from local import Advice
from constant import DEBUG
from parse import find_var_domain
from result import *


class Rule:
    results = {}
    codes = []
    Advice = None

    def __init__(self, debug='INFO', no=1, advice='避免调用运行时间过长的接口'):
        self.debug = debug
        self.no = no
        self.advice = advice

    def __call__(self, func):

        def decorate(*args, **kwargs):
            old, new = func(*args, **kwargs)
            if old is not None:
                if DEBUG == 'INFO':
                    self.info(old, new)
                elif DEBUG == 'RESULT':
                    self.result(old, new)
                elif DEBUG == 'ALL':
                    self.info(old, new)
                    self.result(old, new)

        return decorate

    @classmethod
    def init_cls(cls, codes):
        cls.codes = codes

    def _info_(self, origin, new):
        # TODO: Python3.7不支持ast.unparse
        # if new is not None:
        #     lineno = origin.lineno
        #     origin = utils.unparse(origin) if isinstance(origin, ast.AST) else origin
        #
        #     new = utils.unparse(new) if isinstance(new, ast.AST) else new
        #     print(f"[{self.debug}] 第 {lineno} 行优化建议 :{origin} -> {new}")
        # else:  # 没有给出修改建议的话，取第一行输出
        #     unparse = utils.unparse(origin)
        #     print(f"[{self.debug}] 第 {origin.lineno} 行:{unparse}")
        unparse = Rule.codes[origin.lineno-1].strip()
        print(f"[{self.debug}] 第 {origin.lineno} 行:{unparse}")

    def info(self, origin, new):
        print('-' * 100)
        print(f"[{self.debug}] ADVICE{self.no}:{self.advice}")
        if isinstance(origin, list) and isinstance(new, list):
            for i, v in enumerate(origin):
                self._info_(origin[i], new[i])
        else:
            self._info_(origin, new)

    def _result_(self, origin: ast.AST, new):
        # res = {
        #     'Line': origin.lineno,
        #     'Column': origin.col_offset,
        #     # 取问题代码出现的第一行
        #     'Origin': utils.unparse(origin),
        #     # 给出修改结点的/有特殊情况建议/普通建议
        #     'Advice': utils.unparse(new) if isinstance(new, ast.AST) else new if new else self.advice
        # }
        value = [
            origin.lineno,
            origin.col_offset,
            Rule.codes[origin.lineno-1].strip(),
            # utils.unparse(new) if isinstance(new, ast.AST) else new if new else self.advice,
            self.advice,
            self.no
        ]
        res = {'title': self.advice, 'value': value}
        if self.no in Rule.results:
            res_dict = Rule.results[self.no]
            res_dict['value'].append(value)
        else:
            Rule.results[self.no] = res

    def result(self, origin, new):
        if isinstance(origin, list) and isinstance(new, list):
            for i, v in enumerate(origin):
                self._result_(origin[i], new[i])
        else:
            self._result_(origin, new)

    @classmethod
    def convert_results(cls):
        res = Result()
        if Rule.results:
            res.error_code = error_code['success']
            for no in Rule.results:
                extend_res_dict = Rule.results[no]
                extend_res = ExtendResult(extend_res_dict['title'], extend_res_dict['value'])
                res.extend_result.append(extend_res)
        else:
            res.error_code = error_code['optimized']
        return res


@Rule(debug=DEBUG, no=1, advice=Advice.AvoidLongRuntimeInterface.vrec2vdiv)
def vrec2vdiv(node):
    """
    将倒数算子转换为除法
    Args:
        node: 遍历到节点
    Returns: 修改前后的ast节点
    """
    func_name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
    if func_name == 'vrec':
        new_call = ast.Call(
            func=ast.Attribute(value=node.func.value, attr='vdiv', ctx=ast.Load()),
            args=[
                ast.Constant(value=1),
                node.args[0]
            ],
            keywords=[]
        )
        return node, new_call
    else:
        return None, None


@Rule(debug=DEBUG, no=2, advice=Advice.AvoidLongRuntimeInterface.vexp2neg)
def vexp2neg(node):
    """
    存在于分母的指数运算将幂改为负的 1/exp(x) -> exp(-x)
    Args:
        node: 遍历到的ast节点
    Returns: 修改前后的ast节点
    """
    func_name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
    if func_name == 'vdiv' and isinstance(node.args[1], ast.Call) and node.args[1].func.attr == 'vexp':
        inner_call = node.args[1]
        new_call = ast.Call(
            func=ast.Attribute(value=node.func.value, attr='vmul', ctx=ast.Load()),
            args=[
                node.args[0],
                ast.Call(
                    func=ast.Attribute(value=inner_call.func.value, attr='vexp', ctx=ast.Load()),
                    args=[
                        ast.UnaryOp(
                            op=ast.USub(),
                            operand=ast.Name(id=inner_call.args[0].id, ctx=ast.Load())
                        )
                    ],
                    keywords=[]
                )
            ],
            keywords=[]
        )
        return node, new_call
    else:
        return None, None


@Rule(debug=DEBUG, no=3, advice=Advice.AvoidExcessiveWrappingFunction.message)
def excessive_wrap_function(node: ast.FunctionDef):
    # 识别算子实现的主函数，不能把主函数识别为过度封装
    if node.args.args[-1].arg == 'kernel_name':
        return None, None
    name_s = set()
    for n in ast.walk(node):
        for child in ast.iter_child_nodes(n):
            if isinstance(child, ast.Assign):  # 遍历所有赋值语句
                targets = child.targets
                for t in targets:
                    for i in ast.walk(t):
                        # 赋值语句右边可能是元组，还有可能是元组嵌套，所以还需要遍历
                        for c in ast.iter_child_nodes(i):
                            if isinstance(c, ast.Name) and c.id not in name_s:
                                name_s.add(c.id)
    if len(name_s) < 15:
        return node, None
    else:
        return None, None


@Rule(debug=DEBUG, no=4, advice=Advice.InlineConstant.message)
def inline_constant(node):
    # const调用的父结点是一个赋值语句, 如果是，返回这个赋值语句(父结点)
    origin, new = [], []
    func_name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
    if func_name == 'const' and isinstance(node.parent, ast.Assign):
        domain_node = find_var_domain(node)
        origin.append(node.parent)  # 定义的地方
        new.append(Advice.InlineConstant.var_def)
        var_name = node.parent.targets[0].id
        for n in ast.walk(domain_node):
            for c in ast.iter_child_nodes(n):
                # 该变量所有使用的地方
                if isinstance(c, ast.Name) and c.id == var_name and c.parent is not node.parent:
                    origin.append(c)
                    new.append(Advice.InlineConstant.var_use)
        return origin, new
    return None, None


@Rule(debug=DEBUG, no=5, advice=Advice.MultiCore.load_balance)
def multi_core_load_balance(node, soc):
    func_name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
    if func_name == 'for_range':
        block_num_kw = list(filter(lambda x: x.arg == 'block_num', node.keywords))
        if len(block_num_kw) == 0:  # 未设置参数
            return None, None
        block_num = block_num_kw[0].value
        if isinstance(block_num, ast.Constant) and block_num.value % soc['cores'] != 0:
            return node, None
        elif isinstance(block_num, (ast.Name, ast.Call)):
            return node, Advice.MultiCore.check_name
    return None, None


@Rule(debug=DEBUG, no=6, advice=Advice.MultiCore.maximum)
def multi_core_maximum(node, soc):
    func_name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
    if func_name == 'for_range':
        block_num_kw = list(filter(lambda x: x.arg == 'block_num', node.keywords))
        if len(block_num_kw) == 0:  # 未设置参数
            return None, None
        block_num = block_num_kw[0].value
        if isinstance(block_num, ast.Constant) and block_num.value > 65535:
            return node, None
    return None, None


@Rule(debug=DEBUG, no=7, advice=Advice.DoubleBuffer.message)
def double_buffer(node, soc):
    func_name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
    if func_name == 'for_range':
        thread_num_kw = list(filter(lambda x: x.arg == 'thread_num', node.keywords))
        if len(thread_num_kw) != 0:
            return node, None
    return None, None


@Rule(debug=DEBUG, no=8, advice=Advice.SyncInstruction.message)
def sync_instruction(node, soc):
    func_name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
    if func_name == 'new_stmt_scope':
        sync_kw = list(filter(lambda x: x.arg == 'disable_sync', node.keywords))
        if len(sync_kw) != 0 and isinstance(sync_kw[0].value, ast.Constant) and sync_kw[0].value.value:
            return node, None
    return None, None


def get_dsl_rules_in_call():
    return [
        vrec2vdiv,
        vexp2neg,
        inline_constant
    ]


def get_dsl_rules_in_def():
    return [
        excessive_wrap_function
    ]


def get_tik_rules_in_call():
    return [
        multi_core_load_balance,
        multi_core_maximum,
        double_buffer,
        sync_instruction
    ]
