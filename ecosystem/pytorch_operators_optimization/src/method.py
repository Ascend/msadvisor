import re
import os

from util import load_json
import config
from advisor import Advisor

dir_json = load_json(os.path.join(config.CONFIG_PATH, config.OPTIMIZER_DIR))


class TaskOptimizer:
    task_json = dir_json["dir_1"]
    rule = task_json["rule"]
    advice = task_json["advice"]

    @Advisor.register_file_processor('.py')
    @staticmethod
    def run_file(advisor, path, content):
        result = []
        for index in range(len(content)):
            line = content[index]
            if line.lstrip().startswith('#'):
                continue
            for r in TaskOptimizer.rule:
                ret = re.findall(TaskOptimizer.rule[r], line)
                if ret == []:
                    continue
                result.append([
                    path,
                    str(index + 1),
                    line.strip(),
                    TaskOptimizer.advice[r]
                ])
        advisor.result_general.extend(result)


class TaskAmp:
    task_json = dir_json["dir_2"]
    rule = task_json["rule"]
    advice = task_json["advice"]

    @Advisor.register_file_processor('.py')
    @staticmethod
    def run_file(advisor, path, content):
        result = []
        for index in range(len(content)):
            line = content[index]
            if line.lstrip().startswith('#'):
                continue
            ret = re.findall(TaskAmp.rule["_1"], line)  # 这一条匹配很精确，缺点是返回结果只有')'或''
            if ret == []:
                continue
            if ret[0] == ')':
                # 下标123分别加入行号，代码段，建议
                result.append([
                    path,
                    str(index + 1),
                    line.strip(),
                    TaskAmp.advice["_1"]
                ])
        advisor.result_general.extend(result)


class TaskOperators:
    task_json = dir_json["dir_3"]
    rule = task_json["rule"]
    advice = task_json["advice"]

    @Advisor.register_file_processor('.py')
    @staticmethod
    def run_file(advisor, path, content):
        result = []
        for index in range(len(content)):
            line = content[index]
            if line.lstrip().startswith('#'):
                continue
            for r in TaskOperators.rule:
                ret = re.findall(TaskOperators.rule[r], line)
                if ret == []:
                    continue
                result.append([
                    path,
                    str(index + 1),
                    line.strip(),
                    TaskOperators.advice[r]
                ])
        advisor.result_general.extend(result)


class TaskParemeter:
    task_json = dir_json["dir_4"]
    rule1 = task_json["rule1"]
    rule2 = task_json["rule2"]
    advice = task_json["advice"]

    @Advisor.register_file_processor('.sh')
    @staticmethod
    def run_file(advisor, path, content):
        for index in range(len(content)):
            line = content[index]
            if line.lstrip().startswith('#'):
                continue
            for r in TaskParemeter.rule1:
                ret1 = re.findall(TaskParemeter.rule1[r], line)
                if ret1 != []:
                    advisor.vars['lr'] = ret1
                    advisor.vars['lr_path'] = path
                    advisor.vars['lr_line'] = str(index + 1)
                    advisor.vars['lr_anchor'] = line.strip()

                ret2 = re.findall(TaskParemeter.rule2[r], line)
                if ret2 != []:
                    advisor.vars['bs'] = ret2
                    advisor.vars['bs_path'] = path
                    advisor.vars['bs_line'] = str(index + 1)
                    advisor.vars['bs_anchor'] = line.strip()

    @Advisor.register_final_processor
    @staticmethod
    def run_final(advisor):
        ret = []
        if 'lr_path' in advisor.vars or 'bs_path' in advisor.vars:
            ret.append(advisor.vars.get('lr_path', ''))
            ret.append(advisor.vars.get('lr_line', ''))
            ret.append(advisor.vars.get('lr_anchor', ''))
            ret.append(advisor.vars.get('bs_path', ''))
            ret.append(advisor.vars.get('bs_line', ''))
            ret.append(advisor.vars.get('bs_anchor', ''))
            ret.append(TaskParemeter.advice['_1'])
            advisor.result_lrbs.append(ret)


class TaskTasket:
    task_json = dir_json["dir_5"]
    rule = task_json["rule"]
    advice = task_json["advice"]

    @Advisor.register_file_processor('.sh')
    @staticmethod
    def run_file(advisor, path, content):
        for index in range(len(content)):
            line = content[index]
            if line.lstrip().startswith('#'):
                continue
            for r in TaskTasket.rule:
                ret = re.findall(TaskTasket.rule[r], line)
                if ret != []:
                    advisor.vars['file_path'] = path

    @Advisor.register_final_processor
    @staticmethod
    def run_final(advisor):
        ret = []
        if 'file_path' in advisor.vars:
            ret.append(advisor.vars.get('file_path', ''))
            ret.append(TaskTasket.advice['_1'])
            advisor.result_taskset.append(ret)
