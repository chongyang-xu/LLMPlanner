from llm_planner.templates.template import Template

from datetime import datetime


def post_append_ts(msg):
    msg["content"] += f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    return msg


def pre_append_ts(msg):
    msg["content"] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + msg[
        "content"]
    return msg


def true_or_false(msg):
    return 'r' in msg["content"]


true_block = Template().map(pre_append_ts).done()
false_block = Template().map(post_append_ts).done()
test = Template().input().branch(true_or_false, true_block,
                                 false_block).print().done()

test.start(["1+1=?", "strawberry contains #? 'r'", "your name is"])
