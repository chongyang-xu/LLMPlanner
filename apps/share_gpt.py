import json
import time

from llm_planner.message import Message

from llm_planner.agents.Llama3_8B import Llama3_8B
from llm_planner.agents.miniLLM import MiniLLM

#############
# load data
#############
fname = "/DS/dsg-ml/nobackup/cxu/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json"
with open(fname, 'r') as file:
    data = json.load(file)

###################
# run test
###################
actor_ref = MiniLLM.start(max_token=16)

for chat in data:
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

        msg = Message(prompt=prompt)
        answer = actor_ref.ask(msg)

        print(answer["ret"])
        break

    break

actor_ref.stop()
