from typing import List, Dict, Any

from llm_planner.util import timing
from llm_planner.util import model_type, ModelType

from llm_planner.message import Message
from .service import SingleLLMServe

from cachesaver.async_engine.typedefs import Request
from cachesaver.async_engine.batching import AsyncBatcher
from cachesaver.async_engine.caching import AsyncCacher
from diskcache import Cache
import tempfile
import asyncio


class CachedServing22(SingleLLMServe):

    def __init__(self, p_selector, policy_param_: Dict[str, Any]):

        super().__init__()
        self.p_selector = p_selector
        self.init_done = False

        if self.p_selector.use_cache22:
            cfg = policy_param_.get("cache22", None)
            self.batch_size = cfg.get("batch_size", 4)
            self.cache_dir = cfg.get("cache_dir", tempfile.mkdtemp())

        model_str = policy_param_.get("model", None)
        assert model_str is not None
        self.model_type = model_type(model_str)

    def init_service(self):
        if self.init_done:
            return
        # use openai api on default
        if self.model_type == ModelType.OPENAI:
            self.impl = self.p_selector.services[
                "llm_planner.service.OpenAIServe_API"]
        elif self.model_type == ModelType.ANTHROPIC:
            self.impl = self.p_selector.services[
                "llm_planner.service.AnthropicServe_API"]
        else:
            self.impl = self.p_selector.services["llm_planner.service.HFServe"]

        self.impl.init_service()

        self.cache_impl = Cache(self.cache_dir)
        self.cache = AsyncCacher(cache=self.cache_impl, model=self)
        self.batcher = AsyncBatcher(self.cache, batch_size=self.batch_size)

        self.init_done = True

    @timing
    def work_on(self, q_list: List[Message]):
        self.init_service()

        reqs = [Request(str(q), 1, q.qid, "default") for q in q_list]
        responses = asyncio.run(self.batcher.request(reqs))

        for idx, res in enumerate(responses):
            q_list[idx].response = res

        return q_list

    async def request(self, prompt, num_needed=1):

        q_list = [Message(qid=-1, Message=prompt)]

        ret = self.impl.work_on(q_list)

        resp = [q.response for q in ret]

        return resp
