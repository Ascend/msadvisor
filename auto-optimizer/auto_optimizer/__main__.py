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

import logging
import pathlib
from functools import partial
from typing import List, Union

import click
from click_aliases import ClickAliasedGroup

from auto_optimizer.graph_optimizer.optimizer import GraphOptimizer, InferTestConfig
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern import KnowledgeFactory
from .options import (
    arg_path, arg_input, arg_output,
    opt_optimizer,
    opt_recursive,
    opt_verbose,
    opt_soc,
    opt_device,
    opt_infer_test,
    opt_loop,
    opt_threshold,
)


def optimize_onnx(
    optimizer: GraphOptimizer,
    input_model: pathlib.Path,
    output_model: pathlib.Path,
    infer_test: bool,
    config: InferTestConfig,
) -> List[str]:
    '''Optimize a onnx file and save as a new file.'''
    try:
        graph = OnnxGraph.parse(input_model.as_posix(), add_name_suffix=True)
        optimize_action = partial(optimizer.apply_knowledges_with_infer_test, cfg=config) \
            if infer_test else optimizer.apply_knowledges
        applied_knowledges = optimize_action(graph=graph)
        if applied_knowledges:
            if not output_model.parent.exists():
                output_model.parent.mkdir(parents=True)
            graph.save(output_model.as_posix())
        return applied_knowledges
    except RuntimeError as exc:
        logging.warning('%s optimize failed.', input_model.as_posix())
        logging.warning('exception: %s', exc)
        return []


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
    except RuntimeError as exc:
        logging.warning('%s match failed.', model.as_posix())
        logging.warning('exception: %s', exc)
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
    path: Union[pathlib.Path, bytes],
    optimizer: GraphOptimizer,
    recursive: bool,
    verbose: bool
) -> None:
    path_ = pathlib.Path(path.decode()) if isinstance(path, bytes) else path
    onnx_files = list(path_.rglob('*.onnx') if recursive else path_.glob('*.onnx')) \
        if path_.is_dir() else [path_]
    for onnx_file in onnx_files:
        knowledges = evaluate_onnx(optimizer=optimizer, model=onnx_file, verbose=verbose)
        if not knowledges:
            continue
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
@opt_infer_test
@opt_soc
@opt_device
@opt_loop
@opt_threshold
def command_optimize(
    input_model: pathlib.Path,
    output_model: pathlib.Path,
    optimizer: GraphOptimizer,
    infer_test: bool,
    soc: str,
    device: int,
    loop: int,
    threshold: float,
) -> None:
    # compatibility for click < 8.0
    input_model_ = pathlib.Path(input_model.decode()) if isinstance(input_model, bytes) else input_model
    output_model_ = pathlib.Path(output_model.decode()) if isinstance(output_model, bytes) else output_model
    if input_model_ == output_model_:
        logging.warning('output_model is input_model, refuse to overwrite origin model!')
        return
    config = InferTestConfig(
        converter='atc',
        soc=soc,
        device=device,
        loop=loop,
        threshold=threshold
    )
    applied_knowledges = optimize_onnx(
        optimizer=optimizer,
        input_model=input_model_,
        output_model=output_model_,
        infer_test=infer_test,
        config=config,
    )
    if applied_knowledges:
        print('Optimization success')
        print('Applied knowledges: ')
        for knowledge in applied_knowledges:
            print(f'  {knowledge}')
        print(f'Path: {input_model_} -> {output_model_}')
    else:
        print('Optimization failed.')


if __name__ == "__main__":
    cli()
