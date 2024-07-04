from abc import ABC, abstractmethod
from typing import List

from ..query import Query


class Service(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def work_on(self, q_list: List[Query]):
        pass


class PythonService(Service):
    pass


class KVStoreService(Service):
    pass


class LLMService(Service):
    pass
