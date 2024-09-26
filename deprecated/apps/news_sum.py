import json
import time

from llm_planner.message import Message

from llm_planner.agents.Llama3_8B import Llama3_8B
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
actor_ref = MiniLLM.start(max_token=128)

for news in data:

    start_time = time.time()
    article = news["article"][:200]
    summary = news["summary"]

    PROMPT = f"Provide me a concise summary of this news:\n<news>\n{article}\n</news>"

    msg = Message(prompt=PROMPT)
    answer = actor_ref.ask(msg)
    print(answer["ret"][0])
    break

actor_ref.stop()
