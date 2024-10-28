import os
import requests
from io import BytesIO
from PIL import Image
from llm_planner.message import Message
from llm_planner.actor.system import System
from llm_planner.actor.agent import Agent
from llm_planner.agents.Printer import Printer

class IBMLogoPrinter(Agent):
    def __init__(self):
        super().__init__()
        self.printer = Printer()

    async def process(self, sender_id, message: Message):
        if message["content"] == "print IBM logo":
            # Download IBM logo
            url = "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg"
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            
            # Convert SVG to PNG
            png_path = "ibm_logo.png"
            img.save(png_path, "PNG")
            
            # Prepare message for Printer agent
            print_msg = message.spawn()
            print_msg["content"] = png_path
            print_msg["filename"] = "ibm_logo.pdf"
            
            # Send to Printer agent
            self.send(self.printer.id, print_msg)
        
        elif message.get("pdf_file"):
            print(f"PDF created: {message['pdf_file']}")
            # Clean up the temporary PNG file
            os.remove("ibm_logo.png")

# Set up the system
ibm_logo_printer = IBMLogoPrinter()

# Create and send the initial message
msg = Message()
msg["content"] = "print IBM logo"
ibm_logo_printer.send(ibm_logo_printer.id, msg)

# Start the system
System.start()