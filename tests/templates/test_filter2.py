from llm_planner.message import Message

from llm_planner.templates.template import Template


def ok(msg: Message):
    if "?" in msg["content"]:
        return True
    else:
        return False


test2 = Template().input().filter(ok).ask("mini_llm").print()
test2.start(["1+1=?", "strawberry contains #? 'r'", "your name is"])
