import json

from llm_planner.message import Message
from llm_planner.templates.template import Template


def prepare_data():
    #############
    # load data
    #############
    fname = "/DS/dsg-ml/nobackup/cxu/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json"
    with open(fname, 'r') as file:
        data = json.load(file)
    print("Message preprocessing finish.")
    return data


def prompt(message: Message):
    chat = message["content"]

    id = chat['id']
    conversations = chat['conversations']

    prompt = ""
    msgs = []

    for round in conversations:
        from_who = round['from']
        what_value = round['value']

        prompt += "User: \n" if from_who == "human" else "Assistant: \n"
        prompt += what_value

        if from_who == "gpt":
            # fill content from dataset
            continue

        msg = message.spawn()
        msg["content"] = prompt
        msgs.append(msg)

    return msgs


share_gpt = Template().input().map(prompt).ask("mini_llm").print()
data = prepare_data()
share_gpt.start(data[:4])
