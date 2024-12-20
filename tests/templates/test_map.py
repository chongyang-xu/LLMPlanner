from llm_planner.message import Message

from llm_planner.templates.template import Template

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


test = Template().input().map(update).print()

test.start(["strawberry contains #? 'r'"])
