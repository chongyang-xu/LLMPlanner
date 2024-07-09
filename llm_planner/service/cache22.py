from typing import List, Dict, Any

from llm_planner.util import timing

from llm_planner.query import Query
from .service import CacheService


class Cache22(CacheService):

    def __init__(self, p_selector, policy_param_: Dict[str, Any]):
        pass

    def init_service(self):
        pass

    @timing
    def work_on(self, q_list: List[Query]):
        pass
