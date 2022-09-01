import sys
import argparse

from auto_optimizer.graph_optimizer.optimizer import GraphOptimizer
from auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from auto_optimizer.pattern.knowledges import *

def _opt_parser(opt_parser):
    opt_parser.add_argument('input_model', help='Input model path')
    opt_parser.add_argument('output_model', help='Output model path')
    opt_parser.add_argument('-k', '--knowledge', type=str, metavar='knowledge_names', nargs='+', help='name of knwoledges')

def main():
    parser = argparse.ArgumentParser(description='AutoOptimizer')

    subparsers = parser.add_subparsers(help='commands')
    opt_parser = subparsers.add_parser('opt', help='optimize graph')

    _opt_parser(opt_parser)

    args = parser.parse_args()

    cmd = sys.argv[1]
    if cmd == 'opt':
        knwoledges = args.knowledge
        if knwoledges:
            graph_opt = GraphOptimizer(knwoledges)
            graph = OnnxGraph.parse(args.input_model)
            graph_opt.apply_knowledges(graph)
            graph.save(args.output_model)

if __name__ == "__main__":
    main()