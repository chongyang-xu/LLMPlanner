from llm_planner.templates.template import Template

from datetime import datetime


def integrate(msgs):
    msg = msgs[0].spawn()
    msg["content"] = ""
    for m in msgs:
        msg["content"] += m["content"]
    return msg


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


def append_timestamp(string, format="%Y-%m-%d %H:%M:%S"):
    # Get the current timestamp
    timestamp = datetime.now().strftime(format)
    # Append the timestamp to the string
    return f"{string}\n{timestamp}"


def append_ts(msg):
    msg["content"] = append_timestamp(msg["content"])
    return msg


sub_template = Template().map(update).reduce(integrate).map(append_ts).done()

test = Template().input().repeat(3, sub_template).print().done()

test.start(["strawberry contains #? 'r'"])
