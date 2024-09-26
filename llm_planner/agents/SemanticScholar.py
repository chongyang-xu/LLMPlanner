import os
import requests

from llm_planner.agents.agent import Agent

S2_API_KEY = os.environ["S2_API_KEY"] if "S2_API_KEY" in os.environ else None


class SemanticScholar(Agent):

    def __init__(self):
        super().__init__()
        self.result_limit = 10

    def receive(self, message):

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
