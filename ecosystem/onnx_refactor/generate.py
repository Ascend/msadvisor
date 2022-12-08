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

import os
import json
import stat

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory

# file can write or read, if file not exist, will create
FILE_FLAG = os.O_WRONLY | os.O_CREAT
# file permission, 644
FILE_STAT = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
# current directory path
CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def remove_if_existed(path):
    if os.path.exists(path):
        os.remove(path)
        if os.path.exists(path):
            print('[warning] remove old {} failed.'.format(path))
            return False
    return True

def generate_ecosystem_json():
    model_list = []
    knowledge_pool = KnowledgeFactory.get_knowledge_pool()
    need_extract_knowledge = ['KnowledgeDynamicReshape']
    for knowledge in knowledge_pool:
        adapter_name = knowledge
        knowledge_conf = {'model_name': adapter_name, 'session_list': []}
        knowledge_conf['session_list'] = [{
            'python_model_path': '',
            'parameter': {
                'mode': 'optimize',
                'model_file': '',
                'extract': 1 if knowledge in need_extract_knowledge else 0
            }
        }]
        model_list.append(knowledge_conf)
    if len(model_list) == 0:
        print('[warning] model list is empty.')
        return False
    json_path = '%s/ecosystem.json' % CUR_DIR_PATH
    if not remove_if_existed(json_path):
        return False
    f = os.fdopen(os.open(json_path, FILE_FLAG, FILE_STAT), 'w')
    json.dump({'model_list': model_list}, f, indent=4)
    print('[info] generate ecosystem.json succeed.')
    f.flush()
    f.close()
    os.chmod(json_path, stat.S_IRUSR | stat.S_IRGRP)
    return True

def generate_knowledge_adapter():
    res = False
    knowledge_pool = KnowledgeFactory.get_knowledge_pool()
    for knowledge in knowledge_pool:
        adapter_path = '%s/%s.py' % (CUR_DIR_PATH, knowledge)
        if not remove_if_existed(adapter_path):
            continue
        f = os.fdopen(os.open(adapter_path, FILE_FLAG, FILE_STAT), 'w')
        f.writelines(
            ('# Copyright 2022 Huawei Technologies Co., Ltd' + os.linesep,
            '#' + os.linesep,
            '# Licensed under the Apache License, Version 2.0 (the "License");' + os.linesep,
            '# you may not use this file except in compliance with the License.' + os.linesep,
            '# You may obtain a copy of the License at' + os.linesep,
            '#' + os.linesep,
            '# http://www.apache.org/licenses/LICENSE-2.0' + os.linesep,
            '#' + os.linesep,
            '# Unless required by applicable law or agreed to in writing, software' + os.linesep,
            '# distributed under the License is distributed on an "AS IS" BASIS,' + os.linesep,
            '# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.' + os.linesep,
            '# See the License for the specific language governing permissions and' + os.linesep,
            '# limitations under the License.' + os.linesep,
            '' + os.linesep,
            'import os' + os.linesep,
            'import sys' + os.linesep,
            '' + os.linesep,
            'from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory' + os.linesep,
            '' + os.linesep,
            '' + os.linesep,
            'def evaluate(data_path, param):' + os.linesep,
            '    knowledge = KnowledgeFactory.get_knowledge(\'%s\')' % knowledge + os.linesep,
            '    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))' + os.linesep,
            '    from advisor import evaluate_x' + os.linesep,
            '    return evaluate_x(knowledge, data_path, param)' + os.linesep)
        )
        print('[info] generate %s.py succeed.' % knowledge)
        f.flush()
        f.close()
        res = True
        os.chmod(adapter_path, stat.S_IRUSR | stat.S_IRGRP)
    return res

if __name__ == '__main__':
    if not generate_ecosystem_json():
        print('[error] generate ecosystem.json failed.')
        exit(1)
    if not generate_knowledge_adapter():
        print('[error] generate all knowledge adapter failed.')
        exit(1)
    exit(0)
