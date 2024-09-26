import json
import time

from llm_planner.message import Message
from llm_planner.actor.system import System

from llm_planner.agents.miniLLM import MiniLLM

#############
# load data
#############
news_dataset = "/DS/dsg-ml/nobackup/cxu/datasets/benchmark_llm_summarization/likert_evaluation_results.json"
with open(news_dataset) as file:
    json_data = json.load(file)
data = json_data


###################
# run test
###################
async def main():
    minillm = MiniLLM(max_token=16,
                      return_value=True,
                      with_batching=False,
                      with_caching=True)
    ret = []

    for news in data[:8]:
        article = news["article"][:200]
        summary = news["summary"]

        PROMPT = f"Provide me a concise summary of this news:\n<news>\n{article}\n</news>"

        msg = Message(prompt=PROMPT)
        r = minillm.send(minillm.id, msg)
        ret.append(r)

    await System.finish()

    for r in ret:
        print(await r.value())
        print("-" * 20)


System.start(main)
