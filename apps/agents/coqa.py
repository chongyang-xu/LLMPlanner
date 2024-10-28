import json
import time

from llm_planner.message import Message
from llm_planner.actor.system import System
from llm_planner.actor.agent import Agent

from llm_planner.agents.miniLLM import MiniLLM
from llm_planner.agents.Llama3_8B import Llama3_8B


#############
# load data
#############
def prepare_data():

    fname = "/DS/dsg-ml/nobackup/cxu/datasets/coqa/coqa-dev-v1.0.json"
    with open(fname, 'r') as file:
        json_data = json.load(file)
    data = json_data["data"]
    print("Message preprocessing finish.")
    return data


class CoQAAgent(Agent):

    def __init__(self, data_surce):
        super().__init__()
        self.data = data_surce

        self.sample_round_idx = {}
        self.sample_round = {}
        self.sample_history = {}

        self.minillm = None

    async def process(self, sender_id, message: Message):

        if message["response"] is not None:
            rmsg = message["request_message"]

            sid = rmsg["sample_id"]
            round_idx = rmsg["sample_round_idx"]
            data_idx = rmsg["data_idx"]

            resp = message["response"]
            # print(f"resp = {resp}")
            self.sample_history[sid] = self.sample_history[sid] + [{
                "role": "assistant",
                "content": resp
            }]

            round_idx += 1
            if round_idx < self.sample_round[sid]:
                roundd = self.data[data_idx]["questions"][round_idx]
                input_text = roundd["input_text"]

                self.sample_history[sid] = self.sample_history[sid] + [{
                    "role": "user",
                    "content": input_text
                }]

                # print(f"{id}@{round_idx}: {resp}")

                msg = message.spawn()
                msg["content"] = self.sample_history[sid]
                msg["sample_id"] = sid
                msg["sample_round_idx"] = round_idx
                msg["data_idx"] = data_idx

                self.send(self.minillm.id, msg)
            else:
                print(f"{sid}@{round_idx}")
                for d in self.sample_history[sid]:
                    print(d["role"], ": ", d["content"])

        elif message["content"] is not None:
            if message["content"] == "start":

                self.minillm = MiniLLM(max_token=16,
                                       return_value=False,
                                       with_batching=True,
                                       with_caching=True)

                for data_idx, sample in enumerate(self.data):

                    sid = sample["id"]
                    story = sample["story"]
                    prompt = f"You are a helping assistant to answer user questions according to:\n{story}\n\n"

                    self.sample_history[sid] = [{
                        "role": "system",
                        "content": prompt
                    }]
                    self.sample_round[sid] = len(sample["questions"])
                    self.sample_round_idx[sid] = 0

                    roundd = sample["questions"][0]
                    input_text = roundd["input_text"]

                    self.sample_history[sid] = self.sample_history[sid] + [{
                        "role": "user",
                        "content": input_text
                    }]

                    msg = message.spawn()
                    msg["content"] = self.sample_history[sid]
                    msg["sample_id"] = sid
                    msg["sample_round_idx"] = 0
                    msg["data_idx"] = data_idx

                    self.send(self.minillm.id, msg)


###################
# run test
###################
data = prepare_data()
coqa = CoQAAgent(data[:2])
msg = Message()
msg['content'] = 'start'
coqa.send(coqa.id, msg)

System.start()
