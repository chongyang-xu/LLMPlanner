import json
import time

from llm_planner.message import Message
from llm_planner.actor.system import System

from llm_planner.agents.miniLLM import MiniLLM

#############
# load data
#############
fname = "/DS/dsg-ml/nobackup/cxu/datasets/coqa/coqa-dev-v1.0.json"
with open(fname, 'r') as file:
    json_data = json.load(file)
data = json_data["data"]


###################
# run test
###################
async def main():
    minillm = MiniLLM(max_token=16,
                      return_value=True,
                      with_batching=False,
                      with_caching=True)
    ret = []

    for sample in data:
        start_time = time.time()
        id = sample["id"]
        story = sample["story"]

        prompt = f"You are a helping assistant to answer user questions according to:\n{story}"

        for round in sample["questions"]:
            input_text = round["input_text"]
            turn_id = round["turn_id"]

            print(turn_id)

            prompt += "User:\n" + input_text

            msg = Message(prompt=prompt)
            r = minillm.send(minillm.id, msg)

            await System.finish()

            resp = await r.value()

            prompt += "Assistant:\n" + resp

            break
        break


System.start(main)
