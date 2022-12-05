import re
import os
import json
import config
from util import *

dir_json = load_json(os.path.join(config.CONFIG_PATH,config.OPTIMIZER_DIR))


class ScanFile:  # 传入参数主路径，递归搜索所有py和sh文件，以路径形式加入list返回
    def __init__(self):
        pass

    def listdir(self, path, list_name):  # 传入存储的list
        lst = os.listdir(path)
        for file in lst:
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                self.listdir(file_path, list_name)
            if file.endswith('.py') or file.endswith('.sh'):
                list_name.append(file_path)


        return list_name

    def run(self, path):
        filepath = []
        result = self.listdir(path, filepath)
        return result


class task_base:
    def __init__(self, advisor) -> None:
        advisor.register_final_processor(self.run_final)

    ## 这个方法会对每个文件内容进行处理，path为处理的文件地址，content为字符串列表
    def run_file(self, advisor, path, content):
        pass


    ## 这个方法会在最后被调用
    def run_final(self, advisor):
        pass


class task1(task_base):
    def __init__(self, advisor):
        super().__init__(advisor)
        advisor.register_file_processor('.py', self.run_file)
        self.task_json=dir_json["dir_1"]
        self.rule=self.task_json["rule"]
        self.advice=self.task_json["advice"]

    def run_file(self, advisor, path, content):
        result = []
        for index in range(len(content)):
            line = content[index]
            if line.lstrip().startswith('#'):
                continue;
            for r in self.rule:
                ret = re.findall(self.rule[r], line)
                if ret == []:
                    continue
                #下标123分别加入行号，代码段，建议
                ret[0]="path:"+path # 将匹配内容替换为path
                ret.append("line:"+str(index+1))
                ret.append("anchor:"+line.strip())
                ret.append("advice:"+self.advice[r])#巧妙利用r的对应关系
                if ret !=[]:
                    result.append(ret)
        if result != []:
            advisor.result_lst.append(result)

        return result


class task2(task_base):
    def __init__(self, advisor):
        super().__init__(advisor)
        advisor.register_file_processor('.py', self.run_file)
        self.task_json=dir_json["dir_2"]
        self.rule=self.task_json["rule"]
        self.advice=self.task_json["advice"]

    def run_file(self, advisor, path, content):
        result = []
        for index in range(len(content)):
            line = content[index]
            if line.lstrip().startswith('#'):
                continue;
            ret = re.findall(self.rule["_1"], line)#这一条匹配很精确，缺点是返回结果只有')'或''
            if ret == []:
                continue
            if ret[0]==')' :
                #下标123分别加入行号，代码段，建议
                ret[0]="path:"+path# 将匹配内容替换为path
                ret.append("line:"+str(index+1))
                ret.append("anchor:"+line.strip())
                ret.append("advice:"+self.advice["_1"])
                if ret !=[]:
                    result.append(ret)

        if result != []:
            advisor.result_lst.append(result)

        return result

class task3(task_base):
    def __init__(self, advisor):
        super().__init__(advisor)
        advisor.register_file_processor('.py', self.run_file)
        self.task_json=dir_json["dir_3"]
        self.rule=self.task_json["rule"]
        self.advice=self.task_json["advice"]

    def run_file(self, advisor, path, content):
        result = []
        for index in range(len(content)):
            line = content[index]
            if line.lstrip().startswith('#'):
                continue;
            for r in self.rule:
                ret = re.findall(self.rule[r], line)
                if ret == []:
                    continue
                #下标123分别加入行号，代码段，建议
                ret[0]="path:"+path # 将匹配内容替换为path
                ret.append("line:"+str(index+1))
                ret.append("anchor:"+line.strip())
                ret.append("advice:"+self.advice[r])#巧妙利用r的对应关系
                if ret !=[]:
                    result.append(ret)

        if result != []:
            advisor.result_lst.append(result)
        return result

class task4(task_base):
    def __init__(self, advisor):
        super().__init__(advisor)
        advisor.register_file_processor('.sh', self.run_file)
        self.task_json=dir_json["dir_4"]
        self.rule1=self.task_json["rule1"]
        self.rule2=self.task_json["rule2"]
        self.advice=self.task_json["advice"]

    def run_file(self, advisor, path, content):
        for index in range(len(content)):
            line = content[index]
            if line.lstrip().startswith('#'):
                continue;
            for r in self.rule1:
                ret1 = re.findall(self.rule1[r], line)
                if ret1 != []:
                    advisor.vars['lr'] = ret1
                    advisor.vars['lr_path'] = path
                    advisor.vars['lr_line'] = index + 1
                    advisor.vars['lr_anchor'] = line.strip()

                ret2 = re.findall(self.rule2[r], line)
                if ret2 != []:
                    advisor.vars['bs'] = ret2
                    advisor.vars['bs_path'] = path
                    advisor.vars['bs_line'] = index + 1
                    advisor.vars['bs_anchor'] = line.strip()

    def run_final(self, advisor):
        result = []
        ret = []
        if 'lr_path' in advisor.vars:
          ret.append(f"lr path: {advisor.vars['lr_path']}")
          ret.append(f"lr line: {advisor.vars['lr_line']}")
          ret.append(f"lr anchor: {advisor.vars['lr_anchor']}")
          ret.append(f"bs path: {advisor.vars['bs_path']}")
          ret.append(f"bs line: {advisor.vars['bs_line']}")
          ret.append(f"bs anchor: {advisor.vars['bs_anchor']}")
          ret.append(f"advice: {self.advice['_1']}")
        if ret !=[]:
            result.append(ret)
        if result != []:
            advisor.result_lst.append(result)

class task5(task_base):
    def __init__(self, advisor):
        super().__init__(advisor)
        advisor.register_file_processor('.sh', self.run_file)
        self.task_json=dir_json["dir_5"]
        self.rule=self.task_json["rule"]
        self.advice=self.task_json["advice"]

    def run_file(self, advisor, path, content):
        for index in range(len(content)):
            line = content[index]
            for r in self.rule:
                ret = re.findall(self.rule[r], line)
                if ret != []:
                    advisor.vars['temp'] = 1
                advisor.vars['file_path'] = path


    def run_final(self, advisor):
        result = []
        ret = []
        if not 'temp' in advisor.vars:
            if 'file_path' in advisor.vars:
                ret.append(f"lr path: {advisor.vars['file_path']}")
                ret.append(f"advice: {self.advice['_1']}")
        if ret !=[]:
            result.append(ret)
        if result != []:
            advisor.result_lst.append(result)
