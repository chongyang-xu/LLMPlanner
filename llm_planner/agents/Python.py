from llm_planner.actor.agent import Agent
from llm_planner.message import Message

import subprocess
import sys
import os


class Python(Agent):

    async def process(self, sender_id, message: Message):
        content = message["content"]
        msg = message.spawn()
        msg["request_message"] = message

        try:
            # Use subprocess to execute the code within an exec() call
            result = subprocess.run(
                [sys.executable, "-c", f"exec({repr(content)})"],
                env=os.environ,
                capture_output=True,
                text=True,
                timeout=10)

            # print("out: ", result.stdout)
            # print("err: ", result.stderr)
            if result.returncode != 0 or result.stderr != '':
                msg["response"] = f"Error: {result.stderr.strip()}"
            elif result.stdout.strip() != 'Done':
                msg["response"] = f"Sucessful run should print Done but get: {result.stdout.strip()}"
            else:
                msg["response"] = "ok"

        except subprocess.TimeoutExpired:
            msg["response"] = "Error: Execution timed out"
        except Exception as e:
            # Print the output and errors to the terminal
            print(f"Codes:\n{content}")
            print(f"Output:\n{result.stdout}")
            print(f"Errors:\n{result.stderr}")

            msg["response"] = f"Exception: {str(e)}"

        self.send(sender_id, msg)
        return
