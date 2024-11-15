from llm_planner.message import Message

from llm_planner.templates.template import Template

import random


def update(msg):
    content = msg["content"]
    msgs = []
    for i in range(2):
        m = msg.spawn()
        m["content"] = content + " " + str(i)
        msgs.append(m)
    return msgs


def ok(msg):
    return True
    # return "0" in msg["content"]


def integrate(msgs):
    msg = msgs[0].spawn()
    content = ""
    for m in msgs:
        content = content + m["content"] + "\n"
    msg["content"] = content
    return msg


queries = ["test"]
test = Template()

test.input(queries).map(update).filter(ok).reduce(integrate).print().done()
