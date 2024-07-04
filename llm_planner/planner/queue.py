class Queue:

    def __init__(self):
        self.queue = []

    def enq(self, q):
        self.queue.append(q)

    def deq(self):
        ele = self.queue[0]
        del self.queue[0]
        return ele

    def empty(self):
        return len(self.queue) == 0


class Ingress(Queue):

    def __init__(self):
        super().__init__()


class Egress(Queue):

    def __init__(self):
        super().__init__()
