import json

from llm_planner.message import Message
from llm_planner.templates.template import Template
from llm_planner.templates.template import kv_get, kv_put


def prepare_data():

    fname = "/DS/dsg-ml/nobackup/cxu/datasets/coqa/coqa-dev-v1.0.json"
    with open(fname, 'r') as file:
        json_data = json.load(file)
    data = json_data["data"]
    print("Message preprocessing finish.")
    return data


def sys_prompt(message: Message):
    sample = message["content"]
    sid = sample["id"]

    story = sample["story"]
    sys_prompt = f"You are a helping assistant to answer user questions according to:\n{story}\n\n"

    sample_history = [{"role": "system", "content": sys_prompt}]
    kv_put(f"history-{sid}", sample_history)

    return message


def question(message: Message):
    sample = message["content"]
    tid = message.tid[0]

    sid = sample["id"]
    sample_history = kv_get(f"history-{sid}")

    roundd = sample["questions"][tid]
    input_text = roundd["input_text"]

    sample_history = sample_history + [{"role": "user", "content": input_text}]
    kv_put(f"history-{sid}", sample_history)

    msg = message.spawn()
    msg["content"] = sample_history
    msg["sample"] = sample
    return msg


def finish_round(message: Message):
    rmsg = message["request_message"]
    assert rmsg is not None
    sample = rmsg["sample"]
    assert sample is not None

    sid = sample["id"]
    sample_history = kv_get(f"history-{sid}")
    sample_history = sample_history + [{
        "role": "assistant",
        "content": message["content"]
    }]
    kv_put(f"history-{sid}", sample_history)

    msg = message.spawn()
    msg["content"] = sample
    return msg


def n_times(message: Message):
    sample = message["content"]
    return len(sample["questions"])


def clean(message: Message):
    sample = message["content"]
    sid = sample["id"]
    sample_history = kv_get(f"history-{sid}")
    kv_put(f"history-{sid}", None)

    msg = message.spawn()
    msg["content"] = sample_history[-2:]
    return msg


a_round = Template().map(question).ask("mini_llm").map(finish_round)
coqa_tp = Template().input().map(sys_prompt).repeat(n_times,
                                                    a_round).map(clean).print()

data = prepare_data()
coqa_tp.start(data[:4])
