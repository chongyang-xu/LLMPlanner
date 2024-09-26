from llm_planner.agents.agent_wip import Agent
from llm_planner.message import Message

import asyncio
from concurrent.futures import ThreadPoolExecutor

import time
import os
import requests

S2_API_KEY = os.environ["S2_API_KEY"] if "S2_API_KEY" in os.environ else None


class SemanticScholar(Agent):

    def __init__(self, agent_name):
        super().__init__(agent_name)
        self.result_limit = 10

    async def process(self, message: Message) -> None:

        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(executor, self.blocking_request,
                                          message)
            self.futures[message.id] = future

    def blocking_request(self, message: Message):

        query = message["query"]
        print(f"SemanticScholar:receive: query={query}")

        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers={"X-API-KEY": S2_API_KEY},
            params={
                "query":
                    query,
                "limit":
                    result_limit,
                "fields":
                    "title,authors,venue,year,abstract,citationStyles,citationCount",
            },
        )
        print(f"Response Status Code: {rsp.status_code}")
        print(f"Response Content: {rsp.text[:500]}"
             )  # Print the first 500 characters of the response content
        rsp.raise_for_status()
        results = rsp.json()
        total = results["total"]

        if not total:
            return None
        msg = Message()
        msg["ret"] = results["data"]
        print(msg)

        time.sleep(1.0)

        return msg
