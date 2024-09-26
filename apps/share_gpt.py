import json
import time

from llm_planner.message import Message
from llm_planner.actor.system import System

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


async def main():
    minillm = MiniLLM(max_token=16,
                      return_value=True,
                      with_batching=False,
                      with_caching=True)
    ret = []

    for chat in data[:1]:
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
            r = minillm.send(minillm.id, msg)
            ret.append(r)

            # break

        # break

    await System.finish()

    for r in ret:
        print(await r.value())
        print('-' * 20)


System.start(main)
