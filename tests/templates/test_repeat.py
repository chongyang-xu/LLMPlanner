from llm_planner.templates.template import Template

from datetime import datetime


def append_timestamp(string, format="%Y-%m-%d %H:%M:%S"):
    # Get the current timestamp
    timestamp = datetime.now().strftime(format)
    # Append the timestamp to the string
    return f"{string}\n{timestamp}"


def append_ts(msg):
    msg["content"] = append_timestamp(msg["content"])
    return msg


sub_template = Template().map(append_ts)
test = Template().input().repeat(3, sub_template).print()

test.start(["strawberry contains #? 'r'"])
