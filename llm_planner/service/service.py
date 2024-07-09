from abc import ABC, abstractmethod
from typing import List

from ..query import Query


class Service(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def init_service(self):
        pass

    @abstractmethod
    def work_on(self, q_list: List[Query]):
        pass


class PythonService(Service):
    pass


class CacheService(Service):
    pass


class LLMService(Service):
    pass


class SingleLLMServe(LLMService):
    pass


class SingleLLMFinetune(LLMService):
    pass


class SingleLLMTrain(LLMService):
    pass
