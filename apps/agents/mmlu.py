from datasets import load_dataset

from llm_planner.message import Message
from llm_planner.actor.system import System
from llm_planner.actor.agent import Agent

from llm_planner.agents.miniLLM import MiniLLM


def prepare_data():
    PROMPT_TEMPLATE = """
    Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D]

    Question: {prompt}
    A) {a}
    B) {b}
    C) {c}
    D) {d}

    Answer:
    """

    ###################
    # load dataset
    ###################
    mmlu_subset = load_dataset("cais/mmlu", data_dir="abstract_algebra")['test']
    #mmlu_subset = load_dataset("cais/mmlu", data_dir="all")['test']
    q_list = []
    for i in range(mmlu_subset.num_rows):
        q = PROMPT_TEMPLATE.format(prompt=mmlu_subset['question'][i],
                                   a=mmlu_subset['choices'][i][0],
                                   b=mmlu_subset['choices'][i][1],
                                   c=mmlu_subset['choices'][i][2],
                                   d=mmlu_subset['choices'][i][3])
        q_list.append(q)
    print("Message preprocessing finish.")
    return q_list


class MMLUAgent(Agent):

    def __init__(self, query_list):
        super().__init__()
        self.q_list = query_list

    async def process(self, sender_id, message: Message):

        if message["response"] is not None:
            rid = message["request_id"]
            resp = message["response"]
            print(f"{rid}: {resp}")

        elif message["content"] is not None:
            if message["content"] == "start":

                minillm = MiniLLM(max_token=4,
                                  return_value=True,
                                  with_batching=True,
                                  with_caching=True)

                for q in self.q_list:
                    msg = message.spawn()
                    msg["content"] = q
                    self.send(minillm.id, msg)


###################
# run test
###################
q_list = prepare_data()
mmlu = MMLUAgent(q_list[:16])
msg = Message()
msg['content'] = 'start'
mmlu.send(mmlu.id, msg)

System.start()
