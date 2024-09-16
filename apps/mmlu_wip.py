from datasets import load_dataset

from llm_planner.message import Message
from llm_planner.agents.miniLLM_wip import MiniLLM
from llm_planner.agents.agent_wip import start_agents, read_value

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


###################
# run test
###################
async def main():
    mllm = MiniLLM(agent_name="minillm", max_token=16)
    ret = []
    for q in q_list[:4]:
        msg = Message(prompt=q)
        answer = mllm.ask(msg)
        ret.append(answer)

    for i in ret:
        r = await i.value()
        print(r)


start_agents(main)
