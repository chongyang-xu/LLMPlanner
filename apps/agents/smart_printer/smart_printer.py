from llm_planner.message import Message
from llm_planner.actor.system import System
from llm_planner.actor.agent import Agent

from llm_planner.agents.OpenAI import OpenAI
from llm_planner.agents.Anthropic import Anthropic

from llm_planner.agents.miniLLM import MiniLLM
from llm_planner.agents.Llama3_8B import Llama3_8B

from llm_planner.agents.Python import Python

CODE_PROMPT = """Please help me summarize the function of following Agent Class from python package `llm_planner`

{FILE_CODES}

For each agent, please pay attention to proces() function, and how message is used.
Response the summary for all agents
"""

CODE_PROMPT2 = """Help me with code generation, you can use following agent class

{FILE_CODES}

Here is a breif summary of these agent:

{SUMMRAY}

Here are example apps of using agent Class of `llm_planner`
{APP_EXAMPLES}

With the knowledge above, you can figure out how to print a file by using 
the agent class Printer showed above.

I want you to generate futhur code for a print a pdf for request:
{REQUEST}

Please follow these instructions for code generation:
You can use `llm_planner`, add new code, use other packages to finish the task
You can generate new code to abtain resouces to full fill the request
Make sure no format error for python code, no missing initilization, no missing import
Make sure the python code can run and finish print for the reuqest.
"""

import os
import importlib.util


def extract_first_python_code(text):
    # Try to find the starting index of the first "```python"
    try:
        start_index = text.index("```python") + len("```python")
    except ValueError:
        start_index = 0
    try:
        # Try to find the ending index of the first closing "```" after "```python"
        end_index = text.index("```", start_index)
    except ValueError:
        end_index = -1
    # Extract and return the substring between the start and end indexes
    return text[start_index:end_index].strip()

def find_package_path(package_name):
    # Attempt to find the package by name
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return f"Package '{package_name}' not found."
    else:
        return os.path.dirname(spec.origin)


def read_python_files(directory):
    files_content = ""
    # Iterate through files in the specified directory
    for filename in os.listdir(directory):
        # Skip __init__.py and non-Python files
        if filename.endswith(".py") and filename != "__init__.py":
            file_content = read_single_file(directory, filename)
            # Append filename and its content in the desired format
            files_content += file_content
    return files_content


def read_single_file(directory, filename):
    filepath = os.path.join(directory, filename)
    with open(filepath, 'r') as file:
        file_content = file.read()
    return f"{filepath}\n```python\n{file_content}```\n\n\n"


class SmartPrinter(Agent):

    def __init__(self, oai):
        super().__init__()
        self.oai = oai
        # Specify the directory you want to scan
        package_path = find_package_path("llm_planner")
        directory_path = f"{package_path}/agents"
        self.file_codes = read_python_files(directory_path)
        self.app_examples = read_single_file("..", "sot.py")
        self.app_examples += read_single_file("..", "mmlu.py")
        self.print = Print(oai)

    async def process(self, sender_id, message: Message):

        if message["response"] is not None:
            resp = message["response"]
            request = message["request_message"]["request"]
            code_prompt_2 = CODE_PROMPT2.format(
                FILE_CODES=self.file_codes,
                SUMMRAY=resp,
                REQUEST=request,
                APP_EXAMPLES=self.app_examples,
            )
            msg = message.spawn()
            msg['content'] = code_prompt_2
            self.send(self.print.id, msg)
            # print(f"code_prompt2:{code_prompt_2}")
        else:
            code_prompt = CODE_PROMPT.format(FILE_CODES=self.file_codes)
            msg = message.spawn()
            msg["content"] = [{"role": "user", "content": code_prompt}]
            msg["request"] = message["content"]
            self.send(self.oai.id, msg)
            # print(f"code_prompt:{code_prompt}")


class Executer(Agent):

    def __init__(self):
        super().__init__()
        self.pyinterator = Python()

    async def process(self, sender_id, message: Message):
        if message["response"] is None:
            code = extract_first_python_code(message["content"])
            message["content"] = code
            self.send(self.pyinterator.id, message)
        elif message["response"] == "ok":
            print("Done!")
        else:
            err = message["response"]
            print(f"Something wrong : {err}")


class Print(Agent):

    def __init__(self, oai):
        super().__init__()
        self.oai = oai
        self.exe = Executer()

    async def process(self, sender_id, message: Message):
        if message["response"] is not None:
            code = message["response"]
            msg = message.spawn()
            msg["content"] = code
            self.send(self.exe.id, msg)
        else:
            code_prompt_2 = message["content"]
            msg = message.spawn()
            msg["content"] = [{"role": "user", "content": code_prompt_2}]
            self.send(self.oai.id, msg)


# oai = OpenAI(max_token=512)
oai = Anthropic(max_token=512)
# oai = MiniLLM(max_token=1024)
#oai = Llama3_8B(max_token=1024)
sp = SmartPrinter(oai)

msg = Message()
msg['content'] = "print the logo of IBM"
sp.send(sp.id, msg)

System.start()
