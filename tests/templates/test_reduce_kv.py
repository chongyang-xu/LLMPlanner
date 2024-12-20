from llm_planner.message import Message

from llm_planner.templates.template import Template
from llm_planner.templates.template import kv_get, kv_put

import random


def split_string_n_pieces(s, n):
    # Calculate the approximate length of each piece
    avg_len = len(s) // n
    remainder = len(s) % n
    pieces = []
    start = 0

    for i in range(n):
        # Distribute the remainder across the first pieces
        end = start + avg_len + (1 if i < remainder else 0)
        pieces.append(s[start:end])
        start = end

    return pieces


def update(msg):
    content = msg["content"]
    msgs = []
    pieces = split_string_n_pieces(content, 4)
    for i in range(4):
        m = msg.spawn()
        m["content"] = pieces[i]
        msgs.append(m)
    return msgs


def integrate(msgs):
    count = kv_get("count")
    if count is None:
        kv_put("count", 1)
    else:
        kv_put("count", count + 1)

    count = kv_get("count")

    msg = msgs[0].spawn()
    content = ""
    for m in msgs:
        content += m["content"]
    msg["content"] = content

    num_r = kv_get('r')
    num_r = 0 if num_r is None else num_r
    num_r += content.count('r')
    kv_put('r', num_r)

    if count % 2 == 0:
        print(f"#r: {num_r}")

    return msg


test = Template().input().map(update).reduce(integrate).print()
test.start([
    "strawberry contains #? 'r'",
    "Raindrops dance on the rooftops high, Restless whispers from the sky."
])
