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
    # return True
    return "0" in msg["content"]


test = Template()
test.input(["test"]).map(update).filter(ok).print().start()
