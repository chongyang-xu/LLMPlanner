import json
import time

from llm_planner.message import Message
from llm_planner.actor.system import System
from llm_planner.actor.operator import Operator

from llm_planner.operators.miniLLM import MiniLLM


def prepare_data():
    #############
    # load data
    #############
    news_dataset = "/DS/dsg-ml/nobackup/cxu/datasets/benchmark_llm_summarization/likert_evaluation_results.json"
    with open(news_dataset) as file:
        json_data = json.load(file)
    data = json_data
    print("Message preprocessing finish.")

    return data


class NewsSumAgent(Operator):

    def __init__(self, news_list):
        super().__init__()
        self.data = news_list

    async def process(self, sender_id, message: Message):

        if message["response"] is not None:
            rid = message["request_id"]
            resp = message["response"]
            print(f"{rid}: {resp}")

        elif message["content"] is not None:
            if message["content"] == "start":

                minillm = MiniLLM(max_token=16,
                                  return_value=True,
                                  with_batching=True,
                                  with_caching=True)

                for news in self.data:
                    article = news["article"][:200]
                    summary = news["summary"]

                    PROMPT = f"Provide me a concise summary of this news:\n<news>\n{article}\n</news>"

                    msg = message.spawn()
                    msg["content"] = PROMPT

                    self.send(minillm.id, msg)


###################
# run test
###################
data = prepare_data()
news_sum = NewsSumAgent(data[:4])
msg = Message()
msg['content'] = 'start'
news_sum.send(news_sum.id, msg)

System.start()
