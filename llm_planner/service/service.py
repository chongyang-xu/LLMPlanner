# Copyright (c) 2024-2026 MPI-SWS, Germany
# Author: Chongyang Xu <cxu@mpi-sws.org>

from abc import ABC, abstractmethod
from typing import List

from llm_planner.message import Message


class Service(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def init_service(self):
        pass

    @abstractmethod
    def work_on(self, q_list: List[Message]):
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
