from pandas import json_normalize

import json

from llm_planner.query import Query


def load_coqa_story(n_query=16):
    #pd.set_option('display.max_columns', None)
    coqa_dataset = "/DS/dsg-ml/nobackup/cxu/datasets/coqa/coqa-dev-v1.0.json"
    with open(coqa_dataset) as f:
        coqa_json = json.load(f)
    df = json_normalize(coqa_json['data'], sep='_')
    prompts = df['story'].tolist()

    q_list = []

    for idx, p in enumerate(prompts[:n_query]):
        q = Query(idx, p)
        q_list.append(q)

    return q_list
