from llm_planner.message import Message
from llm_planner.actor.system import System
from llm_planner.actor.operator import Operator

from llm_planner.operators.miniLLM import MiniLLM
from llm_planner.operators.Llama3_8B import Llama3_8B

import os

import inspect


class SoftwareReviser(Operator):

    def __init__(self, llm_agent: Operator):
        super().__init__()
        self.counter = 0
        self.llm_agent = llm_agent
        self.PROMPT_TEMPLATE = """
Codebase:

main.py:
{main_code}

snake.py
{snake_code}

game.py
{game_code}

food.py
{food_code}

Review comments:
{review}

Task: You just implemented ``{file_name}`` Given the code and review comments. Revise ``{file_name}``. Implement all functions and additional functions you need. DO NOT LET ME TO IMPLEMENT ANYTHING!!!!
Make sure your response code is runnable.
Do not response any content in {other_filename1}, {other_filename2} and {other_filename3}. Strictly follow the response format. Do not answer any other content or suggestions.

Response format:

```{revised_code}```
"""

    async def process(self, sender_id, message: Message):
        # print(f"Class: {self.__class__.__name__}")

        if message["response"] is not None:
            file_name = message["request_message"]["file_name"]
            new_code = message["response"]

            directory = "./snake_game_output"
            os.makedirs(directory, exist_ok=True)

            file_path = os.path.join(directory, file_name)

            # Write to the file
            with open(file_path, 'w') as file:
                file.write(new_code)
            print(f"SoftwareReviser wrote {file_path}")

        elif message["review"] is not None:
            prompt = self.PROMPT_TEMPLATE.format(
                main_code=message["main.py"],
                snake_code=message["snake.py"],
                game_code=message["game.py"],
                food_code=message["food.py"],
                review=message["review"],
                file_name=message["file_name"],
                other_filename1=message["other_filename1"],
                other_filename2=message["other_filename2"],
                other_filename3=message["other_filename3"],
                revised_code="{revised_code}")
            msg = message.spawn()
            msg["content"] = prompt
            msg["file_name"] = message["file_name"]

            self.send(self.llm_agent.id, msg)


class SoftwareReviewer(Operator):

    def __init__(self, llm_agent: Operator):
        super().__init__()
        self.llm_agent = llm_agent
        self.codes = {}

        self.PROMPT_TEMPLATE = """
Role: You are an expert code reviewer.
Task:
You review the code given by the expert programmer and share your comments. Do not write your own code.

main.py:
```{main_code}```

snake.py:
```{snake_code}```

game.py:
```{game_code}```

food.py:
```{food_code}```

Comments:
{review}"""

    async def process(self, sender_id, message: Message):
        # print(f"Class: {self.__class__.__name__}")

        if message["response"] is not None:
            reviewer_response = message["response"]

            files = ["main.py", "game.py", "snake.py", "food.py"]
            self.revisers = []
            for idx in range(len(files)):
                reviser = SoftwareReviser(self.llm_agent)
                self.revisers.append(reviser)
                msg = message.spawn()
                msg["main.py"] = self.codes["main.py"]
                msg["snake.py"] = self.codes["snake.py"]
                msg["game.py"] = self.codes["game.py"]
                msg["food.py"] = self.codes["food.py"]
                msg["review"] = reviewer_response

                msg["file_name"] = files[idx]
                msg["other_filename1"] = files[(idx + 1) % 4]
                msg["other_filename2"] = files[(idx + 2) % 4]
                msg["other_filename3"] = files[(idx + 3) % 4]

                self.send(reviser.id, msg)

            self.codes.clear()
        elif message["code"] is not None:

            self.codes[message["file_name"]] = message["code"]

            if len(self.codes) < 4:
                return

            prompt = self.PROMPT_TEMPLATE.format(
                main_code=self.codes["main.py"],
                snake_code=self.codes["snake.py"],
                game_code=self.codes["game.py"],
                food_code=self.codes["food.py"],
                review="{review}")
            msg = message.spawn()
            msg["content"] = prompt
            self.send(self.llm_agent.id, msg)


class SoftwareCoder(Operator):

    def __init__(self, llm_agent_: Operator, reviewer_: Operator):
        super().__init__()
        self.llm_agent = llm_agent_
        self.reviewer = reviewer_
        self.PROMPT_TEMPLATE = """
Role: You are an expert programmer. You implement the APIs given by the system architect.

APIs:
{architect_response}

You only need to implement {file_name}. Implement all functions and additional functions you need. DO NOT LET ME TO IMPLEMENT ANYTHING!!!!
Make sure your response code is runnable.
Do not response any content in {other_filename1}, {other_filename2} and {other_filename3}. Strictly follow the response format. Do not answer any other content or suggestions.

Response format:

```{code}```
"""

    async def process(self, sender_id, message: Message):
        # print(f"Class: {self.__class__.__name__}")

        if message["response"] is not None:
            rmsg = message["request_message"]
            rev_msg = Message({
                "file_name": rmsg["file_name"],
                "code": message["response"]
            })
            self.send(self.reviewer.id, rev_msg)

        elif message["architect_response"] is not None:
            prompt = self.PROMPT_TEMPLATE.format(
                architect_response=message["architect_response"],
                file_name=message["file_name"],
                other_filename1=message["other_filename1"],
                other_filename2=message["other_filename2"],
                other_filename3=message["other_filename3"],
                code="{code}")

            msg = message.spawn()
            msg["content"] = prompt
            msg["file_name"] = message["file_name"]

            self.send(self.llm_agent.id, msg)


class SoftwareArchitect(Operator):

    def __init__(self, llm_agent: Operator):
        super().__init__()
        self.llm_agent = llm_agent
        self.PROMPT_TEMPLATE = """
Role: You are a system architect.

User gives you a task. You design a list of files and design a list of APIs with full function signatures (with functionality as comments) for each file to achieve the task.

Task: Write a cli snake game in python.

Response in the format:

Files:
main.py
game.py
snake.py
food.py
......

APIs:
main.py:
Code:```{main_api}```

game.py:
Code:```{game_api}```

snake.py:
Code:```{snake_api}```

food.py:
Code:```{food_api}```
"""

    async def process(self, sender_id, message: Message):
        # print(f"Class: {self.__class__.__name__}")
        if message["response"] is not None:
            architect_response = message["response"]
            files = ["main.py", "game.py", "snake.py", "food.py"]

            self.reviewer = SoftwareReviewer(self.llm_agent)

            self.coders = []
            for idx in range(len(files)):
                coder = SoftwareCoder(self.llm_agent, self.reviewer)
                self.coders.append(coder)
                msg = message.spawn()
                msg["architect_response"] = architect_response
                msg["file_name"] = files[idx]
                msg["other_filename1"] = files[(idx + 1) % 4]
                msg["other_filename2"] = files[(idx + 2) % 4]
                msg["other_filename3"] = files[(idx + 3) % 4]

                self.send(coder.id, msg)

        elif message["content"] == "start":
            msg = message.spawn()
            msg["content"] = self.PROMPT_TEMPLATE
            self.send(self.llm_agent.id, msg)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###################
# run test
###################

minillm = MiniLLM(max_token=1024)
llm = Llama3_8B(max_token=32)

architect = SoftwareArchitect(llm)

msg = Message()
msg['content'] = "start"
architect.send(architect.id, msg)

System.start()
