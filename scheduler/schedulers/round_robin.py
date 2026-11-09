from schedulers.scheduler import Scheduler
from schedulers.task import Task


# TODO add quantum
class RoundRobin(Scheduler):
    def next(self) -> Task:
        next = self.queue.pop()
        if not next.finished:
            self.queue.append(next)
        return next
