#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

import os
from method import (
    TaskOptimizer, TaskAmp, TaskOperators, TaskParemeter, TaskTasket
)
from advisor import Advisor


def evaluate(data_path, parameter):
    """
    interface function called by msadvisor
    Args:
        data_path: string data_path
        parameter: string parameter
    Returns:
        json string of result info
        result must by ad_result
    """

    # do evaluate work by file data
    if not os.path.exists(data_path):
        print("file or dir:", data_path, "no exist")
        raise FileNotFoundError

    ad = Advisor(data_path)
    ad.run()
    return ad.output()


if __name__ == "__main__":

    project_path = "../data/project/"

    ret = evaluate(project_path, "none")
    print("----------result:----------")
    print(ret)
