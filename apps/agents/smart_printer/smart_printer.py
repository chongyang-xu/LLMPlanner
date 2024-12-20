from llm_planner.message import Message
from llm_planner.actor.system import System
from llm_planner.actor.agent import Agent

from llm_planner.agents.OpenAI import OpenAI
from llm_planner.agents.Anthropic import Anthropic

from llm_planner.agents.miniLLM import MiniLLM
from llm_planner.agents.Llama3_8B import Llama3_8B

from llm_planner.agents.Python import Python

import time

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
If the code finishes task sucessfuly, print word Done.
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
        self.print = Print(oai, self)
        self.max_retry = 3
        self.num_retry = 0

    async def process(self, sender_id, message: Message):

        if message["response"] is None:
            if message["retry"] is None:
                code_prompt = CODE_PROMPT.format(FILE_CODES=self.file_codes)
                msg = message.spawn()
                msg["content"] = [{"role": "user", "content": code_prompt}]
                msg["request"] = message["content"]
                self.send(self.oai.id, msg)
                # print(f"code_prompt:{code_prompt}")
            else:
                assert message["retry"] == True
                self.num_retry += 1
                if self.num_retry > self.max_retry:
                    print("Exit at maximum retrying times")
                else:
                    code = message["code"]
                    err = message["error"]
                    retry_prompt = self.code_prompt_2 + "\n You have tried this code:\n```python\n{code}\n```\n\n This try failed with error: {err}\n"
                    # self.code_prompt_2 = retry_prompt
                    msg = message.spawn()
                    msg["content"] = retry_prompt + "\nFix error and try again:\n"
                    self.send(self.print.id, msg)
        else:
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
            self.code_prompt_2 = code_prompt_2
            self.send(self.print.id, msg)
            # print(f"code_prompt2:{code_prompt_2}")


class Executer(Agent):

    def __init__(self, sp):
        super().__init__()
        self.pyinterator = Python()
        self.smart_printer = sp

    async def process(self, sender_id, message: Message):
        if message["response"] is None:
            code = extract_first_python_code(message["content"])
            message["content"] = code
            with open("./temp/gen.py", 'w') as pycode_f:
                    pycode_f.write(code)
            self.send(self.pyinterator.id, message)
        elif message["response"] == "ok":
            print("Done!")
        else:
            req_msg = message["request_message"]
            assert req_msg is not None
            msg = message.spawn()
            msg["retry"] = True
            msg["code"] = req_msg["content"]
            msg["err"] = message["response"]

            ts = int(time.time())
            with open(f"./temp/gen_{ts}.py", 'w') as pycode_f:
                    pycode_f.write(msg["code"])
            with open(f"./temp/err_{ts}.txt", 'w') as pycode_f:
                    pycode_f.write(msg["err"])

            print(f"Something wrong, retrying...")
            self.send(self.smart_printer.id, msg)

class Print(Agent):

    def __init__(self, oai, sp):
        super().__init__()
        self.oai = oai
        self.exe = Executer(sp)

    async def process(self, sender_id, message: Message):
        if message["response"] is not None:
            code = message["response"]
            msg = message.spawn()
            msg["content"] = code
            self.send(self.exe.id, msg)
        else:
            code_prompt_2 = message["content"]
            # print(code_prompt_2)
            msg = message.spawn()
            msg["content"] = [{"role": "user", "content": code_prompt_2}]
            self.send(self.oai.id, msg)


# oai = OpenAI(max_token=512)
oai = Anthropic(max_token=1024)
# oai = MiniLLM(max_token=1024)
#oai = Llama3_8B(max_token=1024)
sp = SmartPrinter(oai)

msg = Message()
msg['content'] = "print the logo of IBM"
sp.send(sp.id, msg)

System.start()
