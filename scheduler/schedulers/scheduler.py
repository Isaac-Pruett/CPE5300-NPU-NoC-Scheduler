from abc import ABC, abstractmethod

from schedulers.task import Task, TaskList
from speculative import EventTrace


class Scheduler(ABC):
    def __init__(self):
        self.queue: list[Task] = []  # if TaskList, then its MLQ scheduling

    @abstractmethod
    def next(self) -> Task:
        pass

    def admit(self, tasks: TaskList):
        if isinstance(tasks, Task):
            tasks = [tasks]
        self.queue.extend(tasks)

    def admit_trace(self, trace: EventTrace):
        tasks = [Task(event=event, preemptible=False) for event in trace.events]
        self.admit(tasks)
