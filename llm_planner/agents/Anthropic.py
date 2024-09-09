from llm_planner.agents.agent import LLMAgent


class Anthropic(LLMAgent):

    def __init__(self):
        super().__init__()

    def receive(self, message):
        return super().receive(message)
