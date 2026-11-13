from schedulers.scheduler import Scheduler
from schedulers.task import Task


class FIFO(Scheduler):
    def next(self) -> Task:
        next = self.queue.pop(0)
        if not next.finished:
            self.queue.append(next)
        return next
