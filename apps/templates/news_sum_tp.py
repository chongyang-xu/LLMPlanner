import json

from llm_planner.message import Message
from llm_planner.templates.template import Template


def prepare_data():
    #############
    # load data
    #############
    news_dataset = "/DS/dsg-ml/nobackup/cxu/datasets/benchmark_llm_summarization/likert_evaluation_results.json"
    with open(news_dataset) as file:
        json_data = json.load(file)
    data = json_data
    print("Message preprocessing finish.")

    return data


def prompt(message: Message):
    news = message["content"]
    article = news["article"][:200]
    # summary = news["summary"]

    PROMPT = f"Provide me a concise summary of this news:\n{article}\n"

    msg = message.spawn()
    msg["content"] = PROMPT
    return msg


###################
# run test
###################
news_sum = Template().input().map(prompt).ask("mini_llm").print()

data = prepare_data()
news_sum.start(data[:4])
