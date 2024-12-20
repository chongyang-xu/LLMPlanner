from llm_planner.templates.template import Template

test = Template().input().ask("mini_llm").print()

test.start(["1+1=?", "strawberry contains #? 'r'"])
