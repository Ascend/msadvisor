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

import pathlib

import click
from auto_optimizer.graph_optimizer.optimizer import GraphOptimizer

from auto_optimizer.pattern.knowledge_factory import KnowledgeFactory


def convert_to_graph_optimizer(ctx: click.Context, param: click.Option, value: str) -> GraphOptimizer:
    '''Process and validate knowledges option.'''
    try:
        return GraphOptimizer([v.strip() for v in value.split(',')])
    except Exception:
        raise click.BadParameter('No valid knowledge provided!')


opt_optimizer = click.option(
    '-k',
    '--knowledges',
    'optimizer',
    default=','.join(
        knowledge for knowledge in KnowledgeFactory.get_knowledge_pool().keys()
        if 'fix' not in knowledge.lower()
    ),
    type=str,
    callback=convert_to_graph_optimizer,
    help='Knowledges(index/name) you want to apply. Seperate by comma(,), Default to all except fix knowledges.'
)


opt_verbose = click.option(
    '-v',
    '--verbose',
    'verbose',
    is_flag=True,
    default=False,
    help='show progress in evaluate mode.'
)


opt_recursive = click.option(
    '-r',
    '--recursive',
    'recursive',
    is_flag=True,
    default=False,
    help='Process onnx in a folder recursively if any folder provided as PATH.'
)


arg_output = click.argument(
    'output_model',
    nargs=1,
    type=click.Path(path_type=pathlib.Path)
)


arg_input = click.argument(
    'input_model',
    nargs=1,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=pathlib.Path
    )
)


arg_path = click.argument(
    'path',
    nargs=1,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        path_type=pathlib.Path
    )
)
