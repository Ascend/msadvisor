from abc import ABC, abstractmethod

class BaseGraph(ABC):

    @abstractmethod
    def parse(self, model):
        pass

    @abstractmethod
    def add_placeholder(self, name, dtype, shape, ph_type='input'):
        pass

    @abstractmethod
    def add_initializer(self, name, value):
        pass

    @abstractmethod
    def add_node(self, name, op_type, attrs={}, domain=None):
        pass

    @abstractmethod
    def insert_node(self, refer_name, insert_node, refer_io_index=0, mode='after'):
        pass

    @abstractmethod
    def get_nodes(self, op_type):
        pass

    @abstractmethod
    def remove(self, name, maps={0:0}):
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
