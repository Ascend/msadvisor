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

from abc import ABC, abstractmethod

class BaseGraph(ABC):

    @classmethod
    @abstractmethod
    def parse(cls, model):
        pass

    @abstractmethod
    def add_placeholder(self, name, dtype, shape, ph_type='input'):
        pass

    @abstractmethod
    def add_initializer(self, name, value):
        pass

    @abstractmethod
    def add_node(self, name, op_type, attrs=None, domain=None):
        pass

    @abstractmethod
    def insert_node(self, refer_name, insert_node, refer_io_index=0, mode='after'):
        pass

    @abstractmethod
    def get_nodes(self, op_type):
        pass

    @abstractmethod
    def remove(self, name, maps=None):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        pass

    @property
    @abstractmethod
    def inputs(self):
        pass

    @property
    @abstractmethod
    def outputs(self):
        pass

    @abstractmethod
    def save(self, path):
        pass
