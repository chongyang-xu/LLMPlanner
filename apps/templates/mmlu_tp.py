# Copyright (c) 2024-2026 MPI-SWS, Germany
# Author: Chongyang Xu <cxu@mpi-sws.org>

from datasets import load_dataset

from llm_planner.templates.template import Template


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


mmlu = Template().input().ask("mini_llm").print()

q_list = prepare_data()
mmlu.start(q_list[:4])
