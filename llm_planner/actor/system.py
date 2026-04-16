# Copyright (c) 2024-2026 MPI-SWS, Germany
# Author: Chongyang Xu <cxu@mpi-sws.org>

import asyncio
import random


class System:
    __instance = None

    def __new__(cls, scheduling_algorithm=None):
        if cls.__instance is None:
            cls.__instance = super(System, cls).__new__(cls)
            cls.__instance.init(scheduling_algorithm)
        elif scheduling_algorithm is not None:
            cls.__instance.scheduling_algorithm = scheduling_algorithm
        return cls.__instance

    def init(self, scheduling_algorithm=None):
        self.actors = {}
        self.scheduling_algorithm = scheduling_algorithm or self.round_robin
        self.round_robin_index = 0

    def register_actor(self, actor: 'Actor'):
        self.actors[actor.id] = actor

    def send(self, sender_id, recipient_id, message):
        recipient_actor = self.actors.get(recipient_id)
        if recipient_actor:
            recipient_actor.mailbox.put_nowait((sender_id, message))
        else:
            assert False, f"Actor {recipient_id} not found"

    async def run(self):
        keep_running = True
        while keep_running:
            actor = await self.scheduling_algorithm()
            sender_id, msg = await actor.mailbox.get()
            await actor.on_receive(sender_id, msg)
            keep_running = self.should_keep_running()

    def should_keep_running(self):
        for aid, actor in self.actors.items():
            if not actor.mailbox.empty():
                return True

        return False

    async def round_robin(self):
        actor_list = list(self.actors.values())
        while True:
            if len(actor_list) < 1:
                await asyncio.sleep(0)
                actor_list = list(self.actors.values())
                continue
            actor = actor_list[self.round_robin_index % len(actor_list)]
            self.round_robin_index += 1
            if not actor.mailbox.empty():
                return actor
            await asyncio.sleep(0)

    async def random_scheduling(self):
        while True:
            actor_list = [
                actor for actor in self.actors.values()
                if not actor.mailbox.empty()
            ]
            if actor_list:
                actor = random.choice(actor_list)
                return actor
            else:
                await asyncio.sleep(0)

    async def stop(self):
        for name, act in self.actors.items():
            await act.finalize()

    @classmethod
    async def finish(cls):
        try:
            system = cls.__instance
            await system.run()
            await system.stop()
        except KeyboardInterrupt:
            print("Stopping the loop.")

    @classmethod
    def start(cls):
        asyncio.run(cls.finish())

    def run_agent(self, agent, msg):
        try:
            asyncio.run(agent.process(msg))
        except:
            pass
