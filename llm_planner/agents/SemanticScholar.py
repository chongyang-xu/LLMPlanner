from llm_planner.actor.agent import Agent

from llm_planner.message import Message

import os
import aiohttp
import asyncio

S2_API_KEY = os.environ["S2_API_KEY"] if "S2_API_KEY" in os.environ else None


class SemanticScholar(Agent):

    def __init__(self):
        super().__init__()
        self.result_limit = 10

    async def process(self, sender_id, message: Message):
        query = message["query"]

        # request header
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        # Define headers and parameters
        headers = {"X-API-KEY": S2_API_KEY}
        params = {
            "query":
                query,
            "limit":
                self.result_limit,
            "fields":
                "title,authors,venue,year,abstract,citationStyles,citationCount",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url,
                                       headers=headers,
                                       params=params,
                                       timeout=10) as response:
                    if response.status == 200:
                        ret = await response.json()
                        if ret is not None:
                            msg = message.spawn()
                            msg['request_message'] = message
                            msg['response'] = ret
                            self.send(sender_id, msg)
                            await asyncio.sleep(1)
                    else:
                        return f"Error: Received non-200 status code {response.status}"

        except aiohttp.ClientConnectorError:
            return f"Error: Could not connect to {url}"
        except asyncio.TimeoutError:
            return f"Error: Request to {url} timed out"
        except Exception as e:
            return f"Error: An unexpected error occurred: {str(e)}"
