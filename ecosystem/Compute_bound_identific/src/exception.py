#!/usr/bin/env python
# coding=utf-8
"""
Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
"""

class ProfilingPathNotFound(Exception):
    def __init__(self, profiling_path: str, *args: object) -> None:
        super().__init__(*args)
        self.profiling_path = profiling_path

    def __str__(self) -> str:
        return f'Profiling path {self.profiling_path} does not exist'


class KernelMetaNotFound(Exception):
    def __init__(self, kernel_meta_path: str, *args: object) -> None:
        super().__init__(*args)
        self.kernel_meta_path = kernel_meta_path

    def __str__(self) -> str:
        return f'Kernel meta path {self.kernel_meta_path} does not exist'


class ProfilingDataLack(Exception):
    def __init__(self, lack_keys, *args: object) -> None:
        super().__init__(*args)
        self.lack_keys = lack_keys
    
    def __str__(self) -> str:
        return f'Profiling data lack keys: {self.lack_keys}'


class CCEFileNotFound(Exception):
    def __init__(self, cce_path: str, *args: object) -> None:
        super().__init__(*args)
        self.cce_path = cce_path

    def __str__(self) -> str:
        return f'CCE file is missing. Please make sure *.cce file is in {self.cce_path} folder'