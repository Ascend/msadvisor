import csv
import json
import re
import struct
import os
import sys
import getopt
import ge_ir_pb2
import traceback

MAGIC_NUM = b'IMOD'
MODEL_HEAD_SIZE = 256
PARTITION_NUM_SIZE = 4
PARTITION_INFO_SIZE = 12

# msadvisor识别的结果类型
CLASS_TYPE = {'op': '0', 'model': '1'}
ERROR_CODE = {'success': '0', 'optimized': '1'}
EXTEND_TYPE = {'list': '0', 'table': '1', 'sourcedata': '2'}
EXTEND_DATA_TYPE = {'str': '0', 'int': '1', 'double': '2'}

# 融合规则文件路径
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
FUSE_RULES_PATH = CURRENT_DIRECTORY + "/knowledge/CUBE_operator_fuse_rules.csv"


class ExtendResult:
    def __init__(self):
        self.type = '0'
        self.extend_title = ""
        # table type is an array with multiple elements, list type with only one element
        self.data_type = []
        # this field is only used for table type result
        self.key = []
        # table type is a two-dimensional array, list type is a one-dimensional array
        self.value = []


class Result:
    def __init__(self):
        self.class_type = '1'
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


def parse_om_model(string):
    if len(string) < len(MAGIC_NUM) or (string[:len(MAGIC_NUM)] != MAGIC_NUM):
        return None
    string = string[MODEL_HEAD_SIZE:]

    if len(string) < PARTITION_NUM_SIZE:
        return None
    partition_num = struct.unpack('I', string[:PARTITION_NUM_SIZE])[0]
    if partition_num == 0:
        return None
    string = string[PARTITION_NUM_SIZE:]
    partitions = string[partition_num * PARTITION_INFO_SIZE:]

    for idx in range(partition_num):
        partition_info = struct.unpack('iII', string[idx * PARTITION_INFO_SIZE: (idx + 1) * PARTITION_INFO_SIZE])
        ptype, offset, size = partition_info
        if ptype == 0:
            model = ge_ir_pb2.ModelDef()
            model.ParseFromString(partitions[offset: offset + size])
            return model

    return None


# 根据路径获取算子融合规则
def getOpRules(fuse_rule_path):
    fr_list = []
    with open(fuse_rule_path, 'r', encoding='GBK') as fp:
        reader = csv.reader(fp)
        try:
            for x in reader:
                fr_list.append(x)
        except Exception as e:
            print(e)
            print('"' + fuse_rule_path + '"' + ' is not a valid csv file!')
            return None

    for i in range(0, len(fr_list)):
        fr_list[i] = fr_list[i][0].split("+")
    return fr_list


# 树的结点定义
class Node:
    def __init__(self, val):
        self.val = val
        self.nexts = []
        self.end = 0


# 向树中添加结点
def addNode(fr_list, root):
    cur = root
    for i in range(0, len(fr_list)):
        flag = 0
        idx = 0
        for j in range(0, len(cur.nexts)):
            if cur.nexts[j].val == fr_list[i].lower():
                flag = 1
                idx = j
                break
        if flag == 0:
            cur.nexts.append(Node(fr_list[i].lower()))
            cur = cur.nexts[len(cur.nexts) - 1]
        else:
            cur = cur.nexts[idx]
    cur.end += 1


# 建树
def CreateTree(fr_list):
    root = Node("")
    for i in range(0, len(fr_list)):
        addNode(fr_list[i], root)
    return root


FuseOp = []


# 从树中查找可以融合的算子
def search(root, target):
    FuseOp.clear()
    cur = root
    ans = []
    for i in range(0, len(target)):
        idx = i
        search_op_process(cur, target, i, idx, ans)
    for j in range(len(FuseOp)):
        ans.append(FuseOp[j])
    return ans


# 具体查找过程
def search_op_process(cur, target, i, idx, ans):
    if cur.end > 0:
        FuseOp.append([i, idx - 1])
    for j in range(0, len(cur.nexts)):
        if idx >= len(target):
            break
        s = cur.nexts[j].val.split("/")
        for k in range(0, len(s)):
            if (target[idx].lower() == s[k] or
                    ((target[idx].lower()[:4] == 'conv') and (s[k][:4] == 'conv' or s[k] == "depthwise_conv2d"))
                    or (target[idx].lower()[:7] == s[k])):
                search_op_process(cur.nexts[j], target, i, idx + 1, ans)
                break

no_index_op = []
# 将算子名进行拆分
def op_name_split(l):
    pattern = re.compile('_(\d+)_?')
    data = []
    index = []
    for i in range(len(l)):
        cur = l[i]
        temp = re.split('/', cur)
        cur = temp[-1]
        while True:
            result = pattern.search(cur)
            if result == None:
                if cur != "":
                    if cur not in no_index_op:
                        no_index_op.append(cur)
                    data.append(cur)
                    index.append('')
                break
            idx = result.span()
            temp = result.group(1)
            if len(index) > 0 and temp == index[-1]:
                cur = cur[idx[1]:]
                continue
            s = cur[:idx[0]]
            if s == 'MaxPool':
                data.append("Pooling(max)")
            elif "Pool" in s:
                data.append("Pooling")
            elif s != ' ' and s != '':
                data.append(cur[:idx[0]])
            else:
                cur = cur[idx[1]:]
                continue
            index.append(temp)
            cur = cur[idx[1]:]
    return data, index


# 获取可融合的算子和第一个算子的下标
def getOpAndIdx(ans, data, index):
    res = []
    for i in range(len(ans)):
        if len(res) == 0:
            res.append(ans[i])
        elif ans[i][0] > res[-1][1]:
            res.append(ans[i])
        elif (ans[i][1] - ans[i][0]) > (res[-1][1] - res[-1][0]):
            res[-1] = ans[i]

    ans = []
    idx = []
    for i in range(len(res)):
        cur = res[i]
        ans.append(data[cur[0]:cur[1] + 1])
        if index[cur[0]] == '':
            idx.append(index[cur[0] + 1])
        else:
            idx.append(index[cur[0]])
    return ans, idx


# 根据Op类中的原始算子名称对算子列表进行拆分
def getOpName_Orign(Op_list, Fz):
    All_Op = []
    Orign_Op = []
    for i in range(len(Op_list)):
        temp_ = []
        for j in range(len(Op_list[i])):
            temp = re.split('/', Op_list[i][j].name)
            relu_name = temp[-1]
            if relu_name[0:6].lower() == "stride":
                temp_.append("stridedread")
            cur = Op_list[i][j].attr['_datadump_original_op_names'].list.s
            Op_split_Orign = 0
            if len(cur) != 0:
                for k in range(len(cur)):
                    t = str(cur[k])[2:len(str(cur[k])) - 1]
                    try:
                        idx = Fz.index(t)
                    except ValueError:
                        continue
                    else:
                        Op_split_Orign = k
                if Op_split_Orign != 0:
                    Op_split_Orign += 1
                for k in range(Op_split_Orign, len(cur)):
                    t = str(cur[k])[2:len(str(cur[k])) - 1]
                    try:
                        idx = All_Op.index(t)
                    except ValueError:
                        if "bias" in t:
                            continue
                        if ".dequant" in t:
                            temp_.append("AscendDequant")
                        elif ".quant" in t:
                            temp_.append("AscendQuant")
                        else:
                            All_Op.append(t)
                            temp_.append(t)
                    else:
                        continue
            else:
                if ".quant" in Op_list[i][j].name:
                    temp_.append("AscendQuant")
                else:
                    temp_.append(Op_list[i][j].name)
            if relu_name[-5:len(relu_name)].lower() == "write":
                temp_.append("stridedwrite")
        Orign_Op.append(temp_)
    return Orign_Op


def execute(om_path, fuse_rule_path):
    with open(om_path, 'rb') as fd:
        string = fd.read()

    model = parse_om_model(string)

    if model == None:
        print('"'+ om_path + '" is not a valid om model!')
        return

    Op_list = []
    if len(model.graph) == 0:
        print("Can not get operators' information of " + om_path)
        return
    for i in range(len(model.graph[0].op)):
        if model.graph[0].op[i].type != 'Const':
            Op_list.append(model.graph[0].op[i])
    cur = []
    Map = {}
    for i in range(len(Op_list)):
        input_num = 0
        for k in range(len(Op_list[i].input)):
            l = Op_list[i].input[k].split(":")
            if l[0] in Map:
                Map[l[0]] = Map[l[0]] + 1
            else:
                Map[l[0]] = 1
            if "const" not in l[0] and "bias" not in l[0]:
                input_num += 1
        if input_num > 1:
            if Op_list[i].name in Map:
                Map[Op_list[i].name] = Map[Op_list[i].name] + 1
            else:
                Map[Op_list[i].name] = 2
    Fz = []
    for j in Map:
        if Map[j] > 1 or j.split("_")[0] == 'Concat':
            Fz.append(j)

    l = []
    Fz_op = []
    for i in range(len(Op_list)):
        try:
            idx = Fz.index(Op_list[i].name)
        except ValueError:
            l.append(Op_list[i])
        else:
            cur.append(l)
            l = []
            Fz_op.append(Op_list[i])
    cur.append(l)

    ans = []
    if len(Fz) == 0:
        l = []
        for i in range(len(cur)):
            for j in range(len(cur[i])):
                l.append(cur[i][j])
        ans.append(l)
    else:
        idx = 0
        for i in range(len(cur)):
            l = []
            flag = False
            pre = ""
            if idx < len(Fz):
                pre = Fz[idx]
            for k in range(len(cur[i])):
                F = False
                for m in range(len(cur[i][k].input)):
                    input_name = cur[i][k].input[m].split(":")
                    if input_name[0] == pre:
                        F = True
                        l.append(cur[i][k])
                        pre = cur[i][k].name
                        break
                if not F or k == len(cur[i]) - 1:
                    ans.append(l)
                    l = []
                    pre = cur[i][k].name
                    l.append(cur[i][k])
                    if k == len(cur[i]) - 1:
                        ans.append(l)
                        l = []
            idx += 1

    for i in range(len(Fz_op)):
        l = []
        l.append(Fz_op[i])
        ans.append(l)

    Orign_Op = getOpName_Orign(ans, Fz)
    fr_list = getOpRules(fuse_rule_path)
    root = CreateTree(fr_list)

    FusionOp = []
    FirstIdx = []
    for i in range(len(Orign_Op)):
        data, index = op_name_split(Orign_Op[i])
        ans = search(root, data)
        ans, index = getOpAndIdx(ans, data, index)
        for i in range(len(ans)):
            if len(ans[i]) != 0:
                FusionOp.append(ans[i])
                FirstIdx.append(index[i])

    FusionSeq = []
    for i in range(len(FusionOp)):
        pos = FirstIdx[i]
        seq = ""
        for j in range(len(FusionOp[i])):
            if FusionOp[i][j] in no_index_op:
                seq += FusionOp[i][j] + "/"
            else:
                seq += FusionOp[i][j] + '_' + pos + "/"
                pos = int(pos)
                pos += 1
                pos = str(pos)
        seq = seq.strip('/')
        FusionSeq.append(seq)

    return FusionSeq


def evaluate(data_path, parameter):
    data_path += "/project"

    try:
        parameter = json.loads(parameter)
    except Exception as e:
        print(e)
    if isinstance(parameter, dict) != True:
        print("parameter应为dict格式的数据路径")
        sys.exit()

    if not os.path.exists(data_path):
        print(data_path + " is not found.")
        return
    
    if parameter['om_model'][0] == "":
        del parameter['om_model'][0]
        file_name_list = os.listdir(data_path)
        for file in file_name_list:
            if os.path.splitext(file)[1] == ".om":
                parameter['om_model'].append(file)

    if len(parameter['om_model']) == 0:
        print("Can not find om model in " +  data_path)
        return

    om_paths_list = []
    for om_name in parameter['om_model']:
        om_path = data_path + '/' + om_name
        if not os.path.exists(om_path):
            print('"' + om_path + '"' + ' is not found')
            continue
        om_paths_list.append(om_path)

    if not os.path.exists(FUSE_RULES_PATH):
        print('"' + FUSE_RULES_PATH + '"' + " is not found.")
        return
    
    result = Result()
    result.title = "Fusionable om models and their operators lists"
    for om_path in om_paths_list:
        try:
            if not os.path.isfile(om_path):
                continue        
            fusion_seq = execute(om_path, FUSE_RULES_PATH)
            if fusion_seq == None or len(fusion_seq) == 0:
                continue
            else:
                extend_result = ExtendResult()
                extend_result.extend_title = om_path
                extend_result.type = EXTEND_TYPE['list']
                extend_result.data_type.append(EXTEND_DATA_TYPE['str'])
                extend_result.value.extend(fusion_seq)
                result.extend_result.append(extend_result)
        except Exception as e:
            print("\n##########Exception catch##########")
            print("om_path=",om_path)
            traceback.print_exc()

    if len(result.extend_result) == 0:
        result.class_type = CLASS_TYPE['model']
        result.error_code = ERROR_CODE['optimized']
        result.summary = "No operators need to be fused"
    else:
        result.class_type = CLASS_TYPE['model']
        result.error_code = ERROR_CODE['success']
        result.summary = "Some operators need to be fused"

    return result.generate()


# 主函数接口，在本地调试试使用
if __name__ == '__main__':

    data_path = "../data"
    parameter = '{"om_model":[""]}'
    result = evaluate(data_path, parameter)

    print(result)