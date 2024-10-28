import json
import time

from llm_planner.message import Message
from llm_planner.actor.system import System
from llm_planner.actor.agent import Agent

from llm_planner.agents.miniLLM import MiniLLM


def prepare_data():
    #############
    # load data
    #############
    fname = "/DS/dsg-ml/nobackup/cxu/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json"
    with open(fname, 'r') as file:
        data = json.load(file)
    print("Message preprocessing finish.")
    return data


class ShareGPTAgent(Agent):

    def __init__(self, data_surce):
        super().__init__()
        self.data = data_surce

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

                for chat in self.data:
                    id = chat['id']
                    conversations = chat['conversations']

                    prompt = ""
                    for round in conversations:
                        from_who = round['from']
                        what_value = round['value']

                        prompt += "User: \n" if from_who == "human" else "Assistant: \n"
                        prompt += what_value

                        if from_who == "gpt":
                            # fill content from dataset
                            continue

                        msg = message.spawn()
                        msg["content"] = prompt
                        r = self.send(minillm.id, msg)


###################
# run test
###################
data = prepare_data()
share_gpt = ShareGPTAgent(data[:1])
msg = Message()
msg['content'] = 'start'
share_gpt.send(share_gpt.id, msg)

System.start()
