class Query:

    def __init__(self, qid: int, query: str):
        self.qid = qid
        self.query: str = query
        self.response = None
        self.ingress_time = 0.0
        self.egress_time = 0.0

    def __repr__(self):
        return f"{self.query}"


class PromptedQuery(Query):

    def __init__(self, qid: int, prompt: str, query: str):
        super().__init__(qid, query)
        self.prompt = prompt

    def __repr__(self):
        return f"{self.prompt}\n{self.query}"
