from llm_planner.actor.agent import Agent
from llm_planner.message import Message

import subprocess
import sys


class Python(Agent):

    async def process(self, sender_id, message: Message):
        content = message["content"]
        msg = message.spawn()
        msg["request_message"] = message

        try:
            # Use subprocess to execute the code within an exec() call
            result = subprocess.run(
                [sys.executable, "-c", f"exec({repr(content)})"],
                capture_output=True,
                text=True,
                timeout=10)

            if result.returncode != 0:
                msg["response"] = f"Error: {result.stderr.strip()}"
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
