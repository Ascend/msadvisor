# coding=utf-8
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import config

CLASS_LIST = {}


def analysis_cls(command_dict, layer_dict, layer_file):
    """When this file is imported, all AnaCls_x will be added to CLASS_LIST"""
    result = []
    for cls, id in CLASS_LIST.items():
        result.append(cls(command_dict, layer_dict, id, layer_file))
    return result


def rule_init():
    return [], None


def rule_or(expert, layer):
    """
    Args:
        expert:
        layer:

    Returns:
    """
    result = []
    for i in expert.values():
        for j in layer:
            if i == j[1]:
                result.append([j[0], j[2]])
    return result


def rule_and(expert, layer):
    all_layer = []
    for i in layer:
        all_layer.append(i[1])
    num = len(set(expert.values()) & set(all_layer))
    return num == len(set(expert.values()))


class BaseCls:
    def __init__(self):
        self.rule_list = []

    def run(self):
        for result in self.rule_list:
            if result["command"]:
                return result["command"]
            elif result["action"] is None:
                return None


class AnaCls_1(BaseCls):
    def __init__(self, command_dict, layer_dict, command_id, layer_file):
        """

        Args:
            command_dict:
            id:
            layer_dict:
        """
        self.expert_dict = command_dict[command_id]
        self.layer_dict = layer_dict
        self.file = layer_file
        super().__init__()
        self.rule_list.append(self.rule_1())
        self.rule_list.append(self.rule_2())

    def rule_1(self):
        # command, action =  [], None
        command, action = rule_init()
        if not self.file[config.LOG]:
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["log"], self.layer_dict[config.LOG])
        action = True if not rule_or_result != [] else None
        result = {"command": command, "action": action}
        return result

    def rule_2(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["train"], self.layer_dict[config.TRAIN])
        if rule_or_result != []:
            command = ['-', self.expert_dict["command"]["_1"], rule_or_result[0][1]]
        else:
            command = ['-', self.expert_dict["command"]["_2"], ""]
        result = {"command": command, "action": action}
        return result


CLASS_LIST[AnaCls_1] = "id_1"


class AnaCls_2(BaseCls):
    def __init__(self, command_dict, layer_dict, command_id, layer_file):
        self.expert_dict = command_dict[command_id]
        self.layer_dict = layer_dict
        self.file = layer_file
        super().__init__()
        self.rule_list.append(self.rule_1())
        self.rule_list.append(self.rule_2())
        self.line = '-'
        self.file_name = ''

    def rule_1(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["train_1"], self.layer_dict[config.TRAIN])
        if rule_or_result:
            action = True
            self.line = [result[0] for result in rule_or_result]
            self.file_name = [result[1] for result in rule_or_result]
        else:
            action = None
        result = {"command": command, "action": action}
        return result

    def rule_2(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["train_2"], self.layer_dict[config.TRAIN])
        if not rule_or_result:
            command = [self.line[0], self.expert_dict["command"]["_1"], self.file_name[0]]
        result = {"command": command, "action": action}
        return result


CLASS_LIST[AnaCls_2] = "id_2"


class AnaCls_3(BaseCls):
    def __init__(self, command_dict, layer_dict, command_id, layer_file):
        self.expert_dict = command_dict[command_id]
        self.layer_dict = layer_dict
        self.file = layer_file
        super().__init__()
        self.rule_list.append(self.rule_1())
        self.rule_list.append(self.rule_2())

    def rule_1(self):
        command, action = rule_init()
        if not self.file[config.LOG]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["log"], self.layer_dict[config.LOG])
        action = True if not rule_or_result else None
        result = {"command": command, "action": action}
        return result

    def rule_2(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["train"], self.layer_dict[config.TRAIN])
        if rule_or_result:
            command = ['-', self.expert_dict["command"]["_1"], rule_or_result[0][1]]
        result = {"command": command, "action": action}
        return result


CLASS_LIST[AnaCls_3] = "id_3"


class AnaCls_4(BaseCls):
    def __init__(self, command_dict, layer_dict, command_id, layer_file):
        self.expert_dict = command_dict[command_id]
        self.layer_dict = layer_dict
        self.file = layer_file
        super().__init__()
        self.rule_list.append(self.rule_1())
        self.rule_list.append(self.rule_2())

    def rule_1(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["train_1"], self.layer_dict[config.TRAIN])
        action = True if rule_or_result else None
        result = {"command": command, "action": action}
        return result

    def rule_2(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["train_2"], self.layer_dict[config.TRAIN])
        if not rule_or_result:
            command = ['-', self.expert_dict["command"]["_1"], ""]
        result = {"command": command, "action": action}

        return result


CLASS_LIST[AnaCls_4] = "id_4"


class AnaCls_5(BaseCls):
    def __init__(self, command_dict, layer_dict, command_id, layer_file):
        self.expert_dict = command_dict[command_id]
        self.layer_dict = layer_dict
        self.file = layer_file
        super().__init__()
        self.rule_list.append(self.rule_1())
        self.rule_list.append(self.rule_2())
        self.rule_list.append(self.rule_3())

    def rule_1(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["train_1"], self.layer_dict[config.TRAIN])
        action = True if rule_or_result else None
        result = {"command": command, "action": action}

        return result

    def rule_2(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["train_2"], self.layer_dict[config.TRAIN])
        if rule_or_result:
            action = True
        else:
            command = ['-', self.expert_dict["command"]["_1"], ""]
        result = {"command": command, "action": action}

        return result

    def rule_3(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["train_3"], self.layer_dict[config.TRAIN])
        if rule_or_result:
            for i in rule_or(self.expert_dict["train_3"], self.layer_dict[config.TRAIN]):
                command = [i[0], self.expert_dict["command"]["_1"], i[1]]
        result = {"command": command, "action": action}
        return result


CLASS_LIST[AnaCls_5] = "id_5"


class AnaCls_6(BaseCls):
    def __init__(self, command_dict, layer_dict, command_id, layer_file):
        self.expert_dict = command_dict[command_id]
        self.layer_dict = layer_dict
        self.file = layer_file
        super().__init__()
        self.rule_list.append(self.rule_1())
        self.rule_list.append(self.rule_2())

    def rule_1(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["train_1"], self.layer_dict[config.TRAIN])
        action = True if rule_or_result else None
        result = {"command": command, "action": action}
        return result

    def rule_2(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["train_2"], self.layer_dict[config.TRAIN])
        if not rule_or_result:
            command = ['-', self.expert_dict["command"]["_1"], ""]
        result = {"command": command, "action": action}
        return result


CLASS_LIST[AnaCls_6] = "id_6"


class AnaCls_7(BaseCls):
    def __init__(self, command_dict, layer_dict, command_id, layer_file):
        self.expert_dict = command_dict[command_id]
        self.layer_dict = layer_dict
        self.file = layer_file
        super().__init__()
        self.rule_list.append(self.rule_1())
        self.rule_list.append(self.rule_2())

    def rule_1(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["train_1"], self.layer_dict[config.TRAIN])
        action = True if rule_or_result != [] else None
        result = {"command": command, "action": action}

        return result

    def rule_2(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_or_result_1 = rule_or(self.expert_dict["train_2"], self.layer_dict[config.TRAIN])
        rule_or_result_2 = rule_or(self.expert_dict["train_3"], self.layer_dict[config.TRAIN])
        if (not rule_or_result_1) or rule_or_result_2:
            command = ['-', self.expert_dict["command"]["_1"], ""]
        result = {"command": command, "action": action}
        return result


CLASS_LIST[AnaCls_7] = "id_7"


class AnaCls_8(BaseCls):
    def __init__(self, command_dict, layer_dict, command_id, layer_file):
        self.expert_dict = command_dict[command_id]
        self.layer_dict = layer_dict
        self.file = layer_file
        super().__init__()
        self.rule_list.append(self.rule_1())
        self.rule_list.append(self.rule_2())

    def rule_1(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["module_1"], self.layer_dict[config.TRAIN])
        action = True if rule_or_result else None
        result = {"command": command, "action": action}

        return result

    def rule_2(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["module_2"], self.layer_dict[config.TRAIN])
        if rule_or_result:
            command = ['-', self.expert_dict["command"]["_1"], ""]
        result = {"command": command, "action": action}
        return result


CLASS_LIST[AnaCls_8] = "id_8"


class AnaCls_9(BaseCls):
    def __init__(self, command_dict, layer_dict, command_id, layer_file):
        self.expert_dict = command_dict[command_id]
        self.layer_dict = layer_dict
        self.file = layer_file
        super().__init__()
        self.rule_list.append(self.rule_1())
        self.rule_list.append(self.rule_2())

    def rule_1(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["module_1"], self.layer_dict[config.TRAIN])
        if rule_or_result:
            action = None
        else:
            action = True
        result = {"command": command, "action": action}
        return result

    def rule_2(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_and_result = rule_and(self.expert_dict["module_2"], self.layer_dict[config.TRAIN])
        if rule_and_result:
            command = ['-', self.expert_dict["command"]["_1"], ""]
        result = {"command": command, "action": action}
        return result


CLASS_LIST[AnaCls_9] = "id_9"

class AnaCls_10(BaseCls):
    def __init__(self, command_dict, layer_dict, command_id, layer_file):
        self.expert_dict = command_dict[command_id]
        self.layer_dict = layer_dict
        self.file = layer_file
        super().__init__()
        self.rule_list.append(self.rule_1())
        self.rule_list.append(self.rule_2())

    def rule_1(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["train_1"], self.layer_dict[config.TRAIN])
        action = True if rule_or_result else None
        result = {"command": command, "action": action}
        return result

    def rule_2(self):
        command, action = rule_init()
        if not self.file[config.TRAIN]:
            action = None
            result = {"command": command, "action": action}
            return result
        rule_or_result = rule_or(self.expert_dict["train_2"], self.layer_dict[config.TRAIN])
        if not rule_or_result:
            command = ['-', self.expert_dict["command"]["_1"], ""]
        result = {"command": command, "action": action}
        return result


CLASS_LIST[AnaCls_10] = "id_10"

