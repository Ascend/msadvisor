#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import sys
sys.path.append("../")
from test_search_change import test, test2, test3

if __name__ == "__main__":
    test(['tf.data.Dataset'], './data/script', '$DATAreplace$')
    test(['enable_data_pre_proc'], './data/script', '$NRCreplace$')
    test(['allow_mix_precision'], './data/script', '$AMPreplace$')
    test(['set_split_strategy_by'], './data/script', '$NDOWreplace$')
    test(['hcom_parallel'], './data/script', '$HPreplace$')
    for i in ['tf.is_finite(x)', 'tf.debugging.is_finite', 'tf.math.is_finite (x, name=None)',
              'tf.compat.v1.debugging.is_finite', 'tf.compat.v1.is_finite',
              'tf.compat.v1.math.is_finite', 'tf.is_inf()', 'tf.is_nan()']:
        test2([i], './data/script', '$FINreplace$')
    test2(['dropout'], './data/script', '$DROreplace$')
    test2(['def gelu('], './data/script', '$GELUreplace$')
    test2(['mix_compile_mode'], './data/script', '$MCMreplace$')
    test2(['tf.boolean_mask'], './data/script', '$TBMreplace$')
    for i in ['tf.contrib.rnn.BasicLSTMCell', 'tf.nn.rnn_cell.BasicLSTMCell', 'tf.contrib.rnn.LayerNormBasicLSTMCell']:
        test2([i], './data/script', '$TCRBreplace$')
    test2(['loss = "categorical_crossentropy"'], './data/script', '$LCCreplace$')
    test(['npu_distributed_optimizer_wrapper'], './data/script', '$NDOWreplace$')

    test(['Use fusion library value'], './data/log', '$PLOGreplace$')

    test(['precision_mode: allow_mix_precision'], './data/graph', 'precision_mode: force_fp16')
    test2(['op_debug_level'], './data/graph', '$ODLreplace$')
    test2(['iterations_per_loop: 1'], './data/graph', 'iterations_per_loop: 2')
    test(['Dataset'], './data/script', '$DATAreplace$')
    test(['make_one_shot_iterator'], './data/script', '$MOSIreplace$')
    test3()
