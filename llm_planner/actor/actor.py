import asyncio
import uuid

from .system import System


class Actor:

    def __init__(self):
        self.system = System()
        self.id = str(uuid.uuid4())
        self.system.register_actor(self)
        self.mailbox = asyncio.Queue()

    async def on_receive(self, sender_id, message):
        pass

    def send(self, recipient_id, message):
        self.system.send(self.id, recipient_id, message)
