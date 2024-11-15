from llm_planner.templates.template import Template

test = Template().input().print().done()

test.start(["a", "b", "c"])
