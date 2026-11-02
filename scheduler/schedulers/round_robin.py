from schedulers.scheduler import Scheduler
from schedulers.task import Task, TaskList


# TODO add quantum
class RoundRobin(Scheduler):
    def next(self) -> Task:
        next = self.queue.pop()
        self.queue.append(next)
        return next
