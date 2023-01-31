import os

CLASS_TYPE = {'op': '0', 'model': '1'}
ERROR_CODE = {'success': '0', 'optimized': '1'}
SUMMARY = {
    'optimizable': 'The model could be optimizable',
    'optimized': 'The model is well optimized'
}
EXTEND_TYPE = {'list': '0', 'table': '1', 'sourcedata': '2'}
EXTEND_DATA_TYPE = {'str': '0', 'int': '1', 'double': '2'}


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/config/")

OPTIMIZER_DIR = "OptimizerDir.json"
