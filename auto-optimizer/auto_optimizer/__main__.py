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

import sys
import pathlib
from typing import List

import click
from click_aliases import ClickAliasedGroup
from auto_optimizer.graph_optimizer.optimizer import GraphOptimizer

from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern import KnowledgeFactory
from .options import (
    arg_path, arg_input, arg_output, opt_optimizer, opt_recursive, opt_verbose
)


def optimize_onnx(
    optimizer: GraphOptimizer,
    input_model: pathlib.Path,
    output_model: pathlib.Path,
) -> bool:
    '''Optimize a onnx file and save as a new file.'''
    try:
        graph = OnnxGraph.parse(input_model.as_posix(), add_name_suffix=True)
        applied_knowledges = optimizer.apply_knowledges(graph)
        if applied_knowledges:
            if not output_model.parent.exists():
                output_model.parent.mkdir(parents=True)
            graph.save(output_model.as_posix())
            return True
        return False
    except RuntimeError as e:
        print(f'{input_model} optimize failed.', file=sys.stderr)
        print(f'{e}', file=sys.stderr)
        return False


def evaluate_onnx(
    optimizer: GraphOptimizer,
    verbose: bool,
    model: pathlib.Path
) -> List[str]:
    '''Search knowledge pattern in a onnx model.'''
    try:
        if verbose:
            print(f'Evaluating {model.as_posix()}')
        graph = OnnxGraph.parse(model.as_posix(), add_name_suffix=True)
        return optimizer.evaluate_knowledges(graph)
    except RuntimeError as e:
        print(f'{model} match failed.', file=sys.stderr)
        print(f'{e}', file=sys.stderr)
        return []


@click.group(cls=ClickAliasedGroup)
def cli() -> None:
    '''main entrance of auto optimizer.'''
    pass


@cli.command('list', short_help='List available Knowledges.')
def command_list() -> None:
    registered_knowledges = KnowledgeFactory.get_knowledge_pool()
    print('Available knowledges:')
    for idx, name in enumerate(registered_knowledges):
        print(f'  {idx:2d} {name}')


@cli.command(
    'evaluate',
    aliases=['eva'],
    short_help='Evaluate model matching specified knowledges.'
)
@arg_path
@opt_optimizer
@opt_recursive
@opt_verbose
def command_evaluate(
    path: pathlib.Path,
    optimizer: GraphOptimizer,
    recursive: bool,
    verbose: bool
) -> None:
    onnx_files = list(path.rglob('*.onnx') if recursive else path.glob('*.onnx')) \
        if path.is_dir() else [path]
    for onnx_file in onnx_files:
        knowledges = evaluate_onnx(optimizer=optimizer, model=onnx_file, verbose=verbose)
        summary = ','.join(knowledges)
        print(f'{onnx_file}\t{summary}')


@cli.command(
    'optimize',
    aliases=['opt'],
    short_help='Optimize model with specified knowledges.'
)
@arg_input
@arg_output
@opt_optimizer
def command_optimize(
    input_model: pathlib.Path,
    output_model: pathlib.Path,
    optimizer: GraphOptimizer,
) -> None:
    if input_model == output_model:
        print('WARNING: output_model is input_model, refuse to overwrite origin model!')
        return
    if optimize_onnx(optimizer, input_model, output_model):
        print('optimization success')
        print(f'{input_model} -> {output_model}')
    else:
        print('optimization failed.')


if __name__ == "__main__":
    cli()
