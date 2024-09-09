from llm_planner.data.dataset import MiscDataset

import numpy as np
from pandas import json_normalize

import json

from llm_planner.planner.queue import Ingress
from llm_planner.message import Message, PromptedMessage


class CoQARepurpose(MiscDataset):
    pass


def load_coqa_story(n_Message=16):
    #pd.set_option('display.max_columns', None)
    coqa_dataset = "/DS/dsg-ml/nobackup/cxu/datasets/coqa/coqa-dev-v1.0.json"
    with open(coqa_dataset) as f:
        coqa_json = json.load(f)
    df = json_normalize(coqa_json['data'], sep='_')
    prompts = df['story'].tolist()

    q_list = []

    for idx, p in enumerate(prompts[:n_Message]):
        q = Message(idx, p)
        q_list.append(q)

    return q_list


def load_coqa_Message(n_Message=16, shuffle=True):
    #pd.set_option('display.max_columns', None)
    coqa_dataset = "/DS/dsg-ml/nobackup/cxu/datasets/coqa/coqa-dev-v1.0.json"
    with open(coqa_dataset) as f:
        coqa_json = json.load(f)
    df = json_normalize(coqa_json['data'], sep='_')

    idx = 0
    row_idx = 0
    q_list = []

    l_len = []
    q_len = []

    while idx < n_Message:
        story = df.loc[row_idx, 'story']
        qs = df.loc[row_idx, 'questions']

        q_len.append(len(qs))

        for e in qs:
            if not idx < n_Message:
                break
            q_ = PromptedMessage(idx, story, "Question : " + e['input_text'])
            l_len.append(len(str(q_).split(' ')))
            q_list.append(q_)
            idx += 1
        row_idx += 1

    print(f"Length of Message: {np.mean(l_len):.1f}({np.std(l_len):.1f}) words")
    print(f"Number of question: {np.mean(q_len):.1f}({np.std(q_len):.1f})")

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(q_list)

    #for i in range(16):
    #    print(f"len={len(str(q_list[i]))}, {q_list[i].prompt[:20]}, {q_list[i].Message[:10]}")

    return q_list


def load_coqa_Message_to_ingress(n_Message=16, shuffle=True):
    q_list = load_coqa_Message(n_Message=n_Message, shuffle=shuffle)

    ing = Ingress()
    for q in q_list:
        ing.enq(q)

    stop = Message(n_Message, "$stop$")
    ing.enq(stop)

    return ing
