import heapq
from this import s
from typing import override

from schedulers.scheduler import Scheduler
from schedulers.task import Task, TaskList


class CongestionAware(Scheduler):
    @override
    def __init__(self):
        super().__init__()

        self.prio_queue: list[Task] = []

    @override
    def admit(self, tasks: TaskList):
        if isinstance(tasks, Task):
            tasks = [tasks]

        for task in tasks:
            if task.priority is not None:
                heapq.heappush(self.prio_queue, task)
            else:
                self.queue.append(task)

    def next(self) -> Task:
        # prio_queue
        if len(self.prio_queue) != 0:
            next = heapq.heappop(self.prio_queue)
            heapq.heappush(self.prio_queue, next)
            return next

        # non-prio queue
        next = self.queue.pop()
        self.queue.append(next)
        return next
