from enum import Enum
from typing import List
from llm_planner.query import Query


class AllocationType(Enum):
    USER_INPUT = 0
    USER_RESPONSE = 1
    INPUT_DECOMPOSING = 2
    QUERY_ROUTING = 3
    QUERY_GROUPING = 4
    QUERY_REDUCING = 5
    QERTY_SERVING = 6
    NOP = 7


class Allocation:

    def __init__(self,
                 alloc_type: AllocationType,
                 q_list: List[Query],
                 op,
                 eager=False,
                 is_end=False):
        self.type = alloc_type
        self.q_list = q_list
        self.op = op
        self.eager = eager
        self.is_end = is_end
